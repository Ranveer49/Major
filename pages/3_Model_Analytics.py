import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split

from utils.charts import actual_vs_predicted, correlation_heatmap, yield_histogram
from utils.preprocess import FEATURE_COLS, LOG_TRANSFORM, TARGET_H2, VFA_TARGETS, clean_dataframe, generate_synthetic_dataset
from utils.ui import apply_app_style, page_header


st.set_page_config(page_title="Model Analytics", page_icon="📈", layout="wide")
apply_app_style()
page_header(
    "Inspect model quality",
    "Model Analytics",
    "Review hydrogen-yield performance, validation spread, feature importance, and dataset health before using predictions for decisions.",
)


@st.cache_resource
def load_models():
    h_path = "models/hydrogen_model.pkl"
    v_path = "models/vfa_model.pkl"
    h = joblib.load(h_path) if os.path.exists(h_path) else None
    v = joblib.load(v_path) if os.path.exists(v_path) else None
    meta = joblib.load("models/model_meta.pkl") if os.path.exists("models/model_meta.pkl") else {}
    return h, v, meta


@st.cache_data
def load_data():
    path = "dataset/processed_dataset.csv"
    raw = pd.read_csv(path) if os.path.exists(path) else generate_synthetic_dataset()
    return clean_dataframe(raw)


h_model, v_model, meta = load_models()
df = load_data()
if h_model is None:
    st.error("Hydrogen model not found. Run `python train_models.py` first.")
    st.stop()

st.markdown("### Hydrogen Yield Model")
X = df[FEATURE_COLS]
y_raw = df[TARGET_H2].astype(float)
y = np.log1p(y_raw) if LOG_TRANSFORM else y_raw
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

try:
    preds_log = h_model.predict(X_test)
    preds_orig = np.expm1(preds_log) if LOG_TRANSFORM else preds_log
    y_test_orig = np.expm1(y_test) if LOG_TRANSFORM else y_test

    r2 = r2_score(y_test_orig, preds_orig)
    mae = mean_absolute_error(y_test_orig, preds_orig)
    rmse = np.sqrt(mean_squared_error(y_test_orig, preds_orig))
    cv_r2 = meta.get("cv_r2_mean")
    cv_std = meta.get("cv_r2_std")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test R2", f"{r2:.4f}")
    c2.metric("MAE", f"{mae:.4f}")
    c3.metric("RMSE", f"{rmse:.4f}")
    c4.metric("CV R2", f"{cv_r2:.4f} ± {cv_std:.4f}" if cv_r2 is not None else "N/A")

    if r2 >= 0.75:
        st.success(f"R2 = {r2:.4f}. Strong predictive performance for a literature-aggregated bioprocess dataset.")
    elif r2 >= 0.55:
        st.info(f"R2 = {r2:.4f}. Useful performance, with expected limits from dataset heterogeneity.")
    else:
        st.warning(f"R2 = {r2:.4f}. Check data quality, units, and target consistency before relying on predictions.")

    st.plotly_chart(actual_vs_predicted(y_test_orig.values, preds_orig, "Actual vs Predicted - H2 Yield"), use_container_width=True)
except AttributeError as exc:
    st.error("The saved model was trained with a different scikit-learn runtime than the current server.")
    st.info("The repository now pins Python and package versions for Streamlit Cloud. Reboot the app after this fix is deployed.")
    st.caption(f"Model compatibility detail: {exc}")
    st.stop()

with st.expander("Run Live Cross-Validation"):
    if st.button("Run CV"):
        with st.spinner("Running 25-fold evaluation..."):
            rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
            scores = cross_val_score(h_model, X, y, cv=rkf, scoring="r2")
        st.metric("Mean R2", f"{scores.mean():.4f}", delta=f"± {scores.std():.4f}")
        import plotly.express as px

        fig = px.histogram(scores, nbins=15, title="Distribution of CV R2 scores", labels={"value": "R2"})
        st.plotly_chart(fig, use_container_width=True)

st.markdown("### Feature Importance")
try:
    from utils.explainability import shap_feature_importance, shap_summary

    sample = X_train.sample(min(100, len(X_train)), random_state=42)
    with st.spinner("Computing SHAP values..."):
        fig_imp = shap_feature_importance(h_model, sample)
    st.pyplot(fig_imp)

    with st.expander("SHAP Beeswarm Summary"):
        with st.spinner("Building beeswarm..."):
            fig_bee = shap_summary(h_model, sample)
        st.pyplot(fig_bee)
except Exception as exc:
    st.warning(f"SHAP unavailable: {exc}")

if v_model is not None:
    st.markdown("### VFA Model")
    vfa_present = [col for col in VFA_TARGETS if col in df.columns]
    if vfa_present:
        vfa_df = df.dropna(subset=VFA_TARGETS)
        Xv = vfa_df[FEATURE_COLS]
        yv = vfa_df[VFA_TARGETS]
        _, Xv_te, _, yv_te = train_test_split(Xv, yv, test_size=0.2, random_state=42)
        pv = v_model.predict(Xv_te)
        cols = st.columns(len(VFA_TARGETS))
        for i, target in enumerate(VFA_TARGETS):
            cols[i].metric(f"R2 - {target}", f"{r2_score(yv_te.iloc[:, i], pv[:, i]):.4f}")
    else:
        st.info("VFA model uses proxy metabolite text; concentration R2 is not applicable.")

st.markdown("### Dataset Overview")
st.caption(
    f"Rows after cleaning: **{len(df)}** | Yield range: **{df[TARGET_H2].min():.3f} to {df[TARGET_H2].max():.3f}** mol H2/mol substrate"
)

left, right = st.columns(2)
with left:
    st.plotly_chart(yield_histogram(df), use_container_width=True)
with right:
    st.plotly_chart(correlation_heatmap(df), use_container_width=True)
