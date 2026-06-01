import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from utils.preprocess import validate_input

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_xgb_model(pipeline):
    """Extract the XGBRegressor from a sklearn Pipeline."""
    return pipeline.named_steps['model']


def _get_preprocessed(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Run only the preprocessor step of a pipeline."""
    return pipeline.named_steps['preprocessor'].transform(X)


def _feature_names(pipeline, X: pd.DataFrame) -> list[str]:
    """Retrieve feature names after one-hot encoding."""
    prep = pipeline.named_steps['preprocessor']
    try:
        return list(prep.get_feature_names_out())
    except Exception:
        # Fallback: generate generic names
        n = _get_preprocessed(pipeline, X).shape[1]
        return [f"f{i}" for i in range(n)]


# ---------------------------------------------------------------------------
# SHAP waterfall for a single prediction
# ---------------------------------------------------------------------------

def shap_waterfall(pipeline, input_df: pd.DataFrame) -> plt.Figure:
    """
    Generate a SHAP waterfall plot explaining a single prediction.

    Parameters
    ----------
    pipeline : sklearn Pipeline containing 'preprocessor' + 'model'
    input_df : pd.DataFrame — one-row input

    Returns
    -------
    matplotlib Figure
    """
    X = validate_input(input_df.copy())
    X_proc = _get_preprocessed(pipeline, X)
    model = _get_xgb_model(pipeline)
    feat_names = _feature_names(pipeline, X)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_proc)
    shap_values.feature_names = feat_names

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], max_display=15, show=False)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Global feature importance bar chart
# ---------------------------------------------------------------------------

def shap_feature_importance(pipeline, X_train: pd.DataFrame,
                             n_top: int = 15) -> plt.Figure:
    """
    Compute mean |SHAP| values across the training set and return a bar chart.

    Parameters
    ----------
    pipeline : sklearn Pipeline
    X_train  : pd.DataFrame — training features (use a sample for speed)
    n_top    : int — number of top features to display

    Returns
    -------
    matplotlib Figure
    """
    X = validate_input(X_train.copy())
    X_proc = _get_preprocessed(pipeline, X)
    model = _get_xgb_model(pipeline)
    feat_names = _feature_names(pipeline, X)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_proc)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = (
        pd.DataFrame({'Feature': feat_names, 'MeanAbsSHAP': mean_abs_shap})
        .sort_values('MeanAbsSHAP', ascending=False)
        .head(n_top)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        importance_df['Feature'][::-1],
        importance_df['MeanAbsSHAP'][::-1],
        color='steelblue'
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {n_top} Feature Importances (SHAP)")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# SHAP beeswarm summary plot
# ---------------------------------------------------------------------------

def shap_summary(pipeline, X_train: pd.DataFrame) -> plt.Figure:
    """
    SHAP beeswarm plot for an overview of feature impact distribution.
    """
    X = validate_input(X_train.copy())
    X_proc = _get_preprocessed(pipeline, X)
    model = _get_xgb_model(pipeline)
    feat_names = _feature_names(pipeline, X)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_proc)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_proc,
        feature_names=feat_names,
        show=False,
        max_display=15
    )
    plt.tight_layout()
    return fig
