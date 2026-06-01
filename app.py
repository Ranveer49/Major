import streamlit as st

from utils.ui import apply_app_style


st.set_page_config(
    page_title="Biohydrogen Optimization Platform",
    page_icon="🧪",
    layout="wide",
)
apply_app_style()

st.markdown(
    """
    <div class="bio-hero">
        <div class="bio-eyebrow">Dark fermentation decision support</div>
        <h1>Biohydrogen Optimization Platform</h1>
        <p class="bio-lede">
            Predict hydrogen yield, estimate volatile fatty acid profiles, compare
            literature-like experiments, and search operating conditions for stronger
            fermentation performance.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Experimental Runs", "160+")
col2.metric("Substrates", "20+")
col3.metric("Organisms", "35+")
col4.metric("Model Family", "XGBoost")

st.markdown("### Workbench")

left, middle, right = st.columns(3)
with left:
    st.markdown(
        """
        <div class="bio-panel">
            <strong>Prediction Dashboard</strong>
            <p class="bio-small">
                Enter substrate, organism, reactor, temperature, pH, and process
                settings to estimate hydrogen yield and VFA distribution.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with middle:
    st.markdown(
        """
        <div class="bio-panel">
            <strong>Dataset Explorer</strong>
            <p class="bio-small">
                Filter training records, inspect yield distributions, and download
                a focused CSV for analysis or reporting.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        """
        <div class="bio-panel">
            <strong>Process Optimization</strong>
            <p class="bio-small">
                Run a grid search over temperature, pH, and organic loading rate to
                identify high-performing operating windows.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Suggested Flow")
st.markdown(
    """
    <div class="bio-step"><strong>1. Predict:</strong> start with the Prediction Dashboard and enter your planned operating conditions.</div>
    <div class="bio-step"><strong>2. Compare:</strong> use Similar Literature to see which prior experiments are closest to your setup.</div>
    <div class="bio-step"><strong>3. Improve:</strong> use Process Optimization to scan practical temperature, pH, and OLR ranges.</div>
    """,
    unsafe_allow_html=True,
)

st.caption("Models are trained on literature-derived dark fermentation data. Use outputs as decision support, not as a replacement for experimental validation.")
