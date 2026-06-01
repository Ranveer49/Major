"""
utils/predict.py
----------------
Load trained models and run inference.
Handles log-transform inversion automatically via model_meta.pkl.
"""

import os
import joblib
import pandas as pd
import numpy as np

from utils.preprocess import validate_input, VFA_TARGETS

MODEL_DIR = "models"


def _load(filename: str):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Run train_models.py first."
        )
    return joblib.load(path)


def load_hydrogen_model():
    return _load("hydrogen_model.pkl")


def load_vfa_model():
    return _load("vfa_model.pkl")


def load_meta() -> dict:
    path = os.path.join(MODEL_DIR, "model_meta.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return {'log_transform': True}


def predict_hydrogen(model, input_df: pd.DataFrame, meta: dict = None) -> float:
    """
    Predict hydrogen yield for a single input row.
    Automatically inverts log1p if the model was trained with LOG_TRANSFORM.
    """
    if meta is None:
        meta = load_meta()
    X = validate_input(input_df.copy())
    pred = model.predict(X)[0]
    if meta.get('log_transform', True):
        pred = np.expm1(pred)
    return float(np.clip(pred, 0.0, None))


def predict_vfa(model, input_df: pd.DataFrame) -> dict:
    """
    Predict VFA values (concentrations g/L, or proxy presence flags).
    Returns a dict keyed by VFA name.
    """
    X = validate_input(input_df.copy())
    pred = model.predict(X)[0]
    return {
        name: float(np.clip(val, 0.0, None))
        for name, val in zip(VFA_TARGETS, pred)
    }


def predict_all(hydrogen_model, vfa_model, input_df: pd.DataFrame,
                meta: dict = None) -> dict:
    """
    Run both models and return a unified results dict.
    """
    if meta is None:
        meta = load_meta()

    h2   = predict_hydrogen(hydrogen_model, input_df, meta)
    vfas = predict_vfa(vfa_model, input_df) if vfa_model else {
        'Acetate': 0.0, 'Butyrate': 0.0, 'Propionate': 0.0, 'Lactate': 0.0
    }

    dominant_vfa = max(vfas, key=vfas.get)

    # Thauer limit: glucose → 4 mol H₂/mol  (theoretical max)
    theoretical_max = 4.0
    efficiency = min(100.0, (h2 / theoretical_max) * 100)

    pathway_map = {
        'Butyrate':   'Butyrate fermentation — 2 mol H₂/mol glucose (butyric acid route)',
        'Acetate':    'Acetate fermentation — 4 mol H₂/mol glucose (highest theoretical yield)',
        'Propionate': 'Propionate pathway — net H₂ consumer; yield suppressed',
        'Lactate':    'Lactic acid fermentation — minimal H₂ production',
    }

    return {
        'hydrogen_yield': h2,
        'vfas': vfas,
        'dominant_vfa': dominant_vfa,
        'efficiency_score': efficiency,
        'pathway_summary': pathway_map.get(dominant_vfa, 'Mixed fermentation pathway'),
    }
