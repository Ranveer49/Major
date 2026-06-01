import os

import joblib
import numpy as np
import plotly.express as px
import streamlit as st

from utils.optimization import optimization_report, optimize_conditions, sensitivity_analysis
from utils.ui import apply_app_style, page_header


st.set_page_config(page_title="Process Optimization", page_icon="⚙️", layout="wide")
apply_app_style()
page_header(
    "Search operating windows",
    "Process Optimization",
    "Run a grid search across temperature, pH, and organic loading rate to identify high-yield fermentation conditions.",
)


@st.cache_resource
def load_h_model():
    path = "models/hydrogen_model.pkl"
    return joblib.load(path) if os.path.exists(path) else None


h_model = load_h_model()
if h_model is None:
    st.error("Hydrogen model not found. Run `python train_models.py --synthetic` first.")
    st.stop()

st.markdown("### Fixed Conditions")
col1, col2 = st.columns(2)
with col1:
    substrate = st.selectbox("Substrate", ["Glucose", "Sucrose", "Food Waste", "Glycerol", "Starch", "Cellulose", "Lactose", "Whey"])
    organism = st.selectbox("Organism", ["Mixed Culture", "Clostridium butyricum", "Enterobacter aerogenes", "Thermoanaerobacterium"])
    reactor = st.selectbox("Reactor Type", ["Batch", "CSTR", "UASB"])
    culture = st.selectbox("Culture Type", ["MixedCulture", "PureCulture"])
with col2:
    process_mode = st.selectbox("Process Mode", ["Batch", "Continuous"])
    time_h = st.number_input("Incubation Time (h)", value=24.0)
    hrt_h = st.number_input("HRT (h)", value=24.0)
    continuous = st.checkbox("Continuous System", value=process_mode == "Continuous")
    electro = st.checkbox("Electrofermentation")
    nano = st.checkbox("Nanomaterial Assisted")

substrate_map = {
    "Glucose": "Simple_Sugar",
    "Sucrose": "Simple_Sugar",
    "Food Waste": "Organic_Waste",
    "Glycerol": "Industrial_Byproduct",
    "Starch": "Complex_Carbohydrate",
    "Cellulose": "Lignocellulosic",
    "Lactose": "Simple_Sugar",
    "Whey": "Dairy_Byproduct",
}

base_input = {
    "Substrate": substrate,
    "Substrate_Category": substrate_map.get(substrate, "Other"),
    "Organism": organism,
    "Reactor_Type": reactor,
    "Culture_Type": culture,
    "Process_Mode": process_mode,
    "Time_h": time_h,
    "HRT_h": hrt_h,
    "Continuous_System": int(continuous),
    "Thermophilic": 0,
    "Electrofermentation": int(electro),
    "Nanomaterial_Assisted": int(nano),
}

st.markdown("### Optimization Search Ranges")
c1, c2, c3 = st.columns(3)
with c1:
    temp_min = st.number_input("Temp min (°C)", value=30.0)
    temp_max = st.number_input("Temp max (°C)", value=65.0)
    temp_steps = st.slider("Temp steps", 3, 10, 5)
with c2:
    ph_min = st.number_input("pH min", value=5.0)
    ph_max = st.number_input("pH max", value=7.5)
    ph_steps = st.slider("pH steps", 3, 10, 6)
with c3:
    olr_min = st.number_input("OLR min (g/L·d)", value=5.0)
    olr_max = st.number_input("OLR max (g/L·d)", value=25.0)
    olr_steps = st.slider("OLR steps", 3, 10, 5)

top_k = st.slider("Show top N results", 5, 20, 10)

if st.button("Run Optimization", type="primary"):
    temp_range = np.linspace(temp_min, temp_max, temp_steps).tolist()
    ph_range = np.linspace(ph_min, ph_max, ph_steps).tolist()
    olr_range = np.linspace(olr_min, olr_max, olr_steps).tolist()
    total_evals = len(temp_range) * len(ph_range) * len(olr_range)
    st.info(f"Evaluating {total_evals} combinations...")

    with st.spinner("Optimizing..."):
        opt_df = optimize_conditions(h_model, base_input, temp_range=temp_range, ph_range=ph_range, olr_range=olr_range, top_k=top_k)

    st.markdown(optimization_report(opt_df))
    display_cols = ["Temp_C", "pH", "OLR_gL_d", "Predicted_H2_Yield"]
    st.dataframe(
        opt_df[display_cols].rename(
            columns={
                "Temp_C": "Temp (°C)",
                "OLR_gL_d": "OLR (g/L·d)",
                "Predicted_H2_Yield": "H2 Yield (pred.)",
            }
        ),
        use_container_width=True,
    )

    st.markdown("### 3D Parameter Space")
    fig3d = px.scatter_3d(
        opt_df.head(top_k),
        x="Temp_C",
        y="pH",
        z="OLR_gL_d",
        color="Predicted_H2_Yield",
        color_continuous_scale="Viridis",
        labels={"Temp_C": "Temperature (°C)", "OLR_gL_d": "OLR (g/L·d)", "Predicted_H2_Yield": "H2 Yield"},
        title="Top Predicted Conditions in Parameter Space",
    )
    st.plotly_chart(fig3d, use_container_width=True)

    st.markdown("### Sensitivity Analysis")
    best = opt_df.iloc[0]
    base_sens = base_input.copy()
    base_sens["Temp_C"] = best["Temp_C"]
    base_sens["pH"] = best["pH"]
    base_sens["OLR_gL_d"] = best["OLR_gL_d"]
    base_sens["Thermophilic"] = int(best["Temp_C"] > 50)

    param_choice = st.selectbox("Vary parameter", ["Temp_C", "pH", "OLR_gL_d"])
    ranges_map = {
        "Temp_C": np.linspace(20, 80, 30).tolist(),
        "pH": np.linspace(4.0, 9.0, 30).tolist(),
        "OLR_gL_d": np.linspace(1.0, 40.0, 30).tolist(),
    }
    sens_df = sensitivity_analysis(h_model, base_sens, param_choice, ranges_map[param_choice])
    fig_sens = px.line(
        sens_df,
        x=param_choice,
        y="Predicted_H2_Yield",
        title=f"Sensitivity: {param_choice} vs H2 Yield",
        labels={"Predicted_H2_Yield": "Predicted H2 Yield"},
    )
    st.plotly_chart(fig_sens, use_container_width=True)
else:
    st.info("Set search ranges above and click **Run Optimization**.")
