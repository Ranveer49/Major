import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split

from utils.charts import actual_vs_predicted, correlation_heatmap, yield_histogram
from utils.modeling import evaluate_hydrogen_model, load_or_train_models
from utils.preprocess import FEATURE_COLS, TARGET_H2, VFA_TARGETS
from utils.ui import apply_app_style, page_header


st.set_page_config(page_title="Model Analytics", page_icon="📈", layout="wide")
apply_app_style()
page_header(
    "Inspect model quality",
    "Model Analytics",
    "Review hydrogen-yield performance, validation spread, feature importance, and dataset health before using predictions for decisions.",
)


@st.cache_resource(show_spinner="Loading model analytics...")
def load_models():
    return load_or_train_models()


h_model, v_model, meta, df = load_models()

st.markdown("### Hydrogen Yield Model")
metrics = evaluate_hydrogen_model(h_model, df)
X = metrics["X"]
X_train = metrics["X_train"]
y = metrics["y"]
r2 = metrics["r2"]
mae = metrics["mae"]
rmse = metrics["rmse"]
cv_r2 = meta.get("cv_r2_mean")
cv_std = meta.get("cv_r2_std")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Test R2", f"{r2:.4f}")
c2.metric("MAE", f"{mae:.4f}")
c3.metric("RMSE", f"{rmse:.4f}")
c4.metric("CV R2", f"{cv_r2:.4f} ± {cv_std:.4f}" if cv_r2 is not None else "N/A")

if meta.get("runtime_model") != "Saved model":
    st.info(f"Using a Streamlit-compatible runtime model: {meta.get('runtime_model')}.")

if r2 >= 0.75:
    st.success(f"R2 = {r2:.4f}. Strong predictive performance for a literature-aggregated bioprocess dataset.")
elif r2 >= 0.55:
    st.info(f"R2 = {r2:.4f}. Useful performance, with expected limits from dataset heterogeneity.")
else:
    st.warning(f"R2 = {r2:.4f}. Check data quality, units, and target consistency before relying on predictions.")

st.plotly_chart(
    actual_vs_predicted(metrics["y_test_orig"].values, metrics["preds_orig"], "Actual vs Predicted - H2 Yield"),
    use_container_width=True,
)

st.markdown("### Hydrogen Yield Distribution: Before and After Log Transform")
raw_yield = df[[TARGET_H2]].copy()
raw_yield["Log_Transformed_Hydrogen_Yield"] = np.log1p(raw_yield[TARGET_H2].astype(float))

before_col, after_col = st.columns(2)
with before_col:
    fig_raw = px.histogram(
        raw_yield,
        x=TARGET_H2,
        nbins=30,
        title="Before Log Transform",
        labels={TARGET_H2: "Hydrogen Yield"},
        color_discrete_sequence=["#2864a6"],
    )
    fig_raw.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.72)",
        margin=dict(l=20, r=20, t=58, b=35),
    )
    st.plotly_chart(fig_raw, use_container_width=True)

with after_col:
    fig_log = px.histogram(
        raw_yield,
        x="Log_Transformed_Hydrogen_Yield",
        nbins=30,
        title="After Log Transform",
        labels={"Log_Transformed_Hydrogen_Yield": "log1p(Hydrogen Yield)"},
        color_discrete_sequence=["#16856f"],
    )
    fig_log.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.72)",
        margin=dict(l=20, r=20, t=58, b=35),
    )
    st.plotly_chart(fig_log, use_container_width=True)

with st.expander("Run Live Cross-Validation"):
    if st.button("Run CV"):
        with st.spinner("Running cross-validation..."):
            rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
            scores = cross_val_score(h_model, X, y, cv=rkf, scoring="r2")
        st.metric("Mean R2", f"{scores.mean():.4f}", delta=f"± {scores.std():.4f}")
        import plotly.express as px

        fig = px.histogram(scores, nbins=12, title="Distribution of CV R2 scores", labels={"value": "R2"})
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
except Exception:
    st.info("SHAP is optional on Streamlit Cloud. Core model metrics are available above.")

if v_model is not None:
    st.markdown("### VFA Model")
    vfa_present = [col for col in VFA_TARGETS if col in df.columns]
    if len(vfa_present) == len(VFA_TARGETS):
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
