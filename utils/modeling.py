import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from utils.preprocess import FEATURE_COLS, LOG_TRANSFORM, TARGET_H2, VFA_TARGETS, clean_dataframe, create_preprocessor, generate_synthetic_dataset


def load_dataset() -> pd.DataFrame:
    path = "dataset/processed_dataset.csv"
    raw = pd.read_csv(path) if os.path.exists(path) else generate_synthetic_dataset()
    return clean_dataframe(raw)


def _fit_hydrogen_model(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y_raw = df[TARGET_H2].astype(float)
    y = np.log1p(y_raw) if LOG_TRANSFORM else y_raw
    model = Pipeline(
        [
            ("preprocessor", create_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=240,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X, y)

    try:
        cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
        cv_r2_mean = float(scores.mean())
        cv_r2_std = float(scores.std())
    except Exception:
        cv_r2_mean = None
        cv_r2_std = None

    meta = {
        "log_transform": LOG_TRANSFORM,
        "feature_cols": FEATURE_COLS,
        "yield_cap": float(y_raw.max()),
        "cv_r2_mean": cv_r2_mean,
        "cv_r2_std": cv_r2_std,
        "runtime_model": "RandomForestRegressor",
    }
    return model, meta


def _fit_vfa_model(df: pd.DataFrame):
    vfa_present = [col for col in VFA_TARGETS if col in df.columns]
    if len(vfa_present) != len(VFA_TARGETS):
        return None

    vfa_df = df.dropna(subset=VFA_TARGETS)
    if len(vfa_df) < 20:
        return None

    model = Pipeline(
        [
            ("preprocessor", create_preprocessor()),
            (
                "model",
                MultiOutputRegressor(
                    RandomForestRegressor(
                        n_estimators=180,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    )
                ),
            ),
        ]
    )
    model.fit(vfa_df[FEATURE_COLS], vfa_df[VFA_TARGETS].astype(float))
    return model


def train_runtime_models():
    df = load_dataset()
    h_model, meta = _fit_hydrogen_model(df)
    v_model = _fit_vfa_model(df)
    return h_model, v_model, meta, df


def load_or_train_models():
    df = load_dataset()
    try:
        h_model = joblib.load("models/hydrogen_model.pkl")
        v_model = joblib.load("models/vfa_model.pkl") if os.path.exists("models/vfa_model.pkl") else None
        meta = joblib.load("models/model_meta.pkl") if os.path.exists("models/model_meta.pkl") else {"log_transform": True}
        h_model.predict(df[FEATURE_COLS].head(2))
        if v_model is not None:
            v_model.predict(df[FEATURE_COLS].head(2))
        meta["runtime_model"] = meta.get("runtime_model", "Saved model")
        return h_model, v_model, meta, df
    except Exception:
        h_model, meta = _fit_hydrogen_model(df)
        v_model = _fit_vfa_model(df)
        return h_model, v_model, meta, df


def evaluate_hydrogen_model(model, df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y_raw = df[TARGET_H2].astype(float)
    y = np.log1p(y_raw) if LOG_TRANSFORM else y_raw
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preds = model.predict(X_test)
    preds_orig = np.expm1(preds) if LOG_TRANSFORM else preds
    y_test_orig = np.expm1(y_test) if LOG_TRANSFORM else y_test
    return {
        "X": X,
        "X_train": X_train,
        "X_test": X_test,
        "y": y,
        "y_test_orig": y_test_orig,
        "preds_orig": preds_orig,
        "r2": r2_score(y_test_orig, preds_orig),
        "mae": float(np.mean(np.abs(y_test_orig - preds_orig))),
        "rmse": float(np.sqrt(mean_squared_error(y_test_orig, preds_orig))),
    }
