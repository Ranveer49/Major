"""
train_models.py
---------------
Train XGBoost models for:
  1. Hydrogen Yield  (single-output, log-transformed target)
  2. VFA dominant metabolite flags (if VFA numeric data exists)

Outputs
-------
  models/hydrogen_model.pkl
  models/vfa_model.pkl          (or skipped if no VFA data)
  models/label_encoder.pkl      (organism / reactor group maps — for UI)

Usage
-----
  # With your real CSV:
  python train_models.py --csv dataset/processed_dataset.csv

  # Demo (synthetic data):
  python train_models.py --synthetic --n-synthetic 300
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split, cross_val_score, RepeatedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from utils.preprocess import (
    clean_dataframe,
    create_preprocessor,
    generate_synthetic_dataset,
    FEATURE_COLS,
    TARGET_H2,
    VFA_TARGETS,
    LOG_TRANSFORM,
)

os.makedirs("models", exist_ok=True)
os.makedirs("dataset", exist_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train biohydrogen ML models")
    p.add_argument('--csv', default="dataset/processed_dataset.csv")
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--n-synthetic', type=int, default=300)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def print_metrics(name: str, y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bar  = "=" * 42
    print(f"\n{bar}")
    print(f"  {name}")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(bar)
    return r2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame):

    # ── 1. Clean ───────────────────────────────────────────────────────────
    df = clean_dataframe(df)
    print(f"\nRows after cleaning: {len(df)}")
    print(f"Hydrogen_Yield stats:\n{df[TARGET_H2].describe().round(3)}\n")

    # ── 2. Prepare X / y ──────────────────────────────────────────────────
    X = df[FEATURE_COLS]
    y_raw = df[TARGET_H2].astype(float)
    y = np.log1p(y_raw) if LOG_TRANSFORM else y_raw

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── 3. Hydrogen model ─────────────────────────────────────────────────
    preprocessor = create_preprocessor()

    hydrogen_model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(
            n_estimators=600,
            learning_rate=0.025,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=1,
            gamma=0.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        ))
    ])

    hydrogen_model.fit(X_train, y_train)

    # Test-set metrics (log scale)
    preds_log = hydrogen_model.predict(X_test)
    print_metrics("Hydrogen Yield — test set (log scale)", y_test, preds_log)

    # Original-scale metrics
    if LOG_TRANSFORM:
        preds_orig = np.expm1(preds_log)
        y_test_orig = np.expm1(y_test)
        r2_orig = print_metrics(
            "Hydrogen Yield — test set (original scale)",
            y_test_orig, preds_orig
        )

    # Repeated 5-fold CV
    print("\nRunning RepeatedKFold CV (5×5) …")
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    cv_scores = cross_val_score(hydrogen_model, X, y, cv=rkf, scoring='r2')
    print(f"  CV R² (log scale): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    joblib.dump(hydrogen_model, "models/hydrogen_model.pkl")
    print("\n  Saved → models/hydrogen_model.pkl")

    # Save metadata the UI pages need
    meta = {
        'log_transform': LOG_TRANSFORM,
        'feature_cols': FEATURE_COLS,
        'substrate_categories': sorted(df['Substrate_Category'].dropna().unique().tolist()),
        'organism_groups': sorted(df['Organism_Group'].dropna().unique().tolist()),
        'reactor_groups': sorted(df['Reactor_Group'].dropna().unique().tolist()),
        'culture_types': sorted(df['Culture_Type'].dropna().unique().tolist()),
        'yield_cap': float(y_raw.max()),
        'cv_r2_mean': float(cv_scores.mean()),
        'cv_r2_std':  float(cv_scores.std()),
    }
    joblib.dump(meta, "models/model_meta.pkl")
    print("  Saved → models/model_meta.pkl")

    # ── 4. VFA model (only if numeric VFA columns exist) ──────────────────
    vfa_cols_present = [c for c in VFA_TARGETS if c in df.columns]

    if len(vfa_cols_present) == len(VFA_TARGETS):
        vfa_df = df.dropna(subset=VFA_TARGETS)
        if len(vfa_df) >= 20:
            X_vfa  = vfa_df[FEATURE_COLS]
            y_vfa  = vfa_df[VFA_TARGETS].astype(float)

            Xv_tr, Xv_te, yv_tr, yv_te = train_test_split(
                X_vfa, y_vfa, test_size=0.2, random_state=42
            )

            vfa_model = Pipeline([
                ('preprocessor', create_preprocessor()),
                ('model', MultiOutputRegressor(
                    XGBRegressor(
                        n_estimators=400,
                        learning_rate=0.03,
                        max_depth=4,
                        subsample=0.8,
                        colsample_bytree=0.7,
                        min_child_weight=2,
                        random_state=42,
                        verbosity=0,
                    )
                ))
            ])
            vfa_model.fit(Xv_tr, yv_tr)
            preds_vfa = vfa_model.predict(Xv_te)

            for i, t in enumerate(VFA_TARGETS):
                print_metrics(f"VFA — {t}", yv_te.iloc[:, i], preds_vfa[:, i])

            joblib.dump(vfa_model, "models/vfa_model.pkl")
            print("\n  Saved → models/vfa_model.pkl")
            return hydrogen_model, vfa_model
        else:
            print(f"\n  Only {len(vfa_df)} rows with VFA data — need ≥20. Skipping VFA model.")
    else:
        missing = set(VFA_TARGETS) - set(vfa_cols_present)
        print(f"\n VFA columns not found in dataset: {missing}")
        print("    Building proxy VFA model from Metabolites text column …")

        # Proxy: parse Metabolites string to get dominant VFA flags
        if 'Metabolites' in df.columns:
            for vfa in VFA_TARGETS:
                df[vfa] = df['Metabolites'].str.contains(
                    vfa, case=False, na=False
                ).astype(float)

            vfa_df = df.dropna(subset=[TARGET_H2])
            X_vfa = vfa_df[FEATURE_COLS]
            y_vfa = vfa_df[VFA_TARGETS]

            if len(vfa_df) >= 20:
                Xv_tr, Xv_te, yv_tr, yv_te = train_test_split(
                    X_vfa, y_vfa, test_size=0.2, random_state=42
                )
                vfa_model = Pipeline([
                    ('preprocessor', create_preprocessor()),
                    ('model', MultiOutputRegressor(
                        XGBRegressor(n_estimators=300, learning_rate=0.03,
                                     max_depth=4, random_state=42, verbosity=0)
                    ))
                ])
                vfa_model.fit(Xv_tr, yv_tr)
                joblib.dump(vfa_model, "models/vfa_model.pkl")
                print("  Saved proxy VFA model → models/vfa_model.pkl")
                print("    (VFA predictions are presence probabilities, not concentrations)")
                return hydrogen_model, vfa_model

    return hydrogen_model, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if args.synthetic:
        print(f"Generating synthetic dataset ({args.n_synthetic} rows) …")
        df = generate_synthetic_dataset(n_samples=args.n_synthetic)
        df.to_csv("dataset/processed_dataset.csv", index=False)
        print("  Saved → dataset/processed_dataset.csv")
    else:
        if not os.path.exists(args.csv):
            print(f"\n  CSV not found at '{args.csv}'.")
            print("    Run:  python train_models.py --synthetic\n")
            sys.exit(1)
        df = pd.read_csv(args.csv)

    print(f"\nRaw dataset shape : {df.shape}")
    print(f"Columns           : {list(df.columns)}\n")

    h_model, v_model = train(df)

    print("\n  Training complete.")
