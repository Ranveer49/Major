import numpy as np
import pandas as pd
import streamlit as st

from utils.charts import vfa_bar, vfa_pie
from utils.modeling import load_or_train_models
from utils.predict import predict_all
from utils.ui import apply_app_style, page_header


st.set_page_config(page_title="Prediction Dashboard", page_icon="🔬", layout="wide")
apply_app_style()
page_header(
    "Predict fermentation outputs",
    "Hydrogen Yield and VFA Prediction",
    "Tune biological and reactor conditions, then estimate hydrogen yield, dominant VFA pathway, and process efficiency from the trained models.",
)


@st.cache_resource
def load_models():
    h_model, v_model, meta, _ = load_or_train_models()
    return h_model, v_model, meta


hydrogen_model, vfa_model, model_meta = load_models()


SUBSTRATE_CATEGORY = {
    "Glucose": "Simple_Sugar",
    "Sucrose": "Simple_Sugar",
    "Lactose": "Simple_Sugar",
    "Starch": "Complex_Carbohydrate",
    "Cellulose": "Lignocellulosic",
    "Food Waste": "Organic_Waste",
    "Glycerol": "Industrial_Byproduct",
    "Whey": "Dairy_Byproduct",
}
CONTINUOUS_REACTORS = ["CSTR", "UASB", "AnSBR", "Fixed-Bed"]
BATCH_REACTORS = ["Batch", "Serum Bottle"]


st.sidebar.header("Fermentation Parameters")
st.sidebar.markdown("**Biological**")
substrate = st.sidebar.selectbox("Substrate", list(SUBSTRATE_CATEGORY.keys()))
substrate_category = SUBSTRATE_CATEGORY[substrate]
st.sidebar.caption(f"Category auto-set to `{substrate_category}`")

organism = st.sidebar.selectbox(
    "Organism",
    [
        "Mixed Culture",
        "Clostridium butyricum",
        "Clostridium pasteurianum",
        "Enterobacter aerogenes",
        "Thermoanaerobacterium",
        "Caldicellulosiruptor",
        "Anaerobic sludge",
    ],
)
culture = st.sidebar.selectbox("Culture Type", ["MixedCulture", "PureCulture"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Reactor & Process**")
process_mode = st.sidebar.selectbox("Process Mode", ["Batch", "Continuous", "Fed-Batch"])
is_continuous = process_mode == "Continuous"
reactor_options = CONTINUOUS_REACTORS if is_continuous else BATCH_REACTORS + CONTINUOUS_REACTORS
reactor = st.sidebar.selectbox("Reactor Type", reactor_options)

st.sidebar.markdown("---")
st.sidebar.markdown("**Operating Conditions**")
thermophilic = st.sidebar.checkbox("Thermophilic (>50 °C)")
temp_default = 55 if thermophilic else 37
temp_min = 50 if thermophilic else 20
Temp_C = st.sidebar.slider("Temperature (°C)", min_value=temp_min, max_value=80, value=min(temp_default, 80))
pH = st.sidebar.slider("pH", 4.0, 9.0, 6.5, step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Time Parameters**")
if not is_continuous:
    Time_h = st.sidebar.number_input("Incubation Time (h)", value=24.0, min_value=1.0)
    HRT_h = np.nan
    OLR = np.nan
    continuous_flag = 0
else:
    Time_h = np.nan
    continuous_flag = 1
    with st.sidebar.expander("Continuous Reactor Parameters", expanded=True) as exp:
        HRT_h = exp.number_input("HRT - Hydraulic Retention Time (h)", min_value=0.5, value=24.0)
        OLR = exp.number_input("OLR - Organic Loading Rate (g/L·d)", min_value=0.1, value=10.0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Enhancement Technologies**")
electro = st.sidebar.checkbox("Electrofermentation")
if electro:
    with st.sidebar.expander("Electrofermentation Parameters") as exp:
        exp.number_input("Applied Voltage (V)", value=0.5, min_value=0.0)
        exp.selectbox("Electrode Material", ["Carbon cloth", "Graphite", "Stainless steel", "Platinum"])

nano = st.sidebar.checkbox("Nanomaterial Assisted")
if nano:
    with st.sidebar.expander("Nanomaterial Details") as exp:
        exp.selectbox("Nanoparticle Type", ["Fe3O4", "ZnO", "TiO2", "Carbon Nanotubes", "Ag NPs"])
        exp.number_input("Dosage (mg/L)", value=100.0, min_value=0.0)


input_df = pd.DataFrame(
    {
        "Substrate": [substrate],
        "Substrate_Category": [substrate_category],
        "Organism": [organism],
        "Temp_C": [float(Temp_C)],
        "pH": [float(pH)],
        "Time_h": [Time_h],
        "HRT_h": [HRT_h],
        "OLR_gL_d": [OLR],
        "Reactor_Type": [reactor],
        "Culture_Type": [culture],
        "Process_Mode": [process_mode],
        "Continuous_System": [continuous_flag],
        "Thermophilic": [int(thermophilic)],
        "Electrofermentation": [int(electro)],
        "Nanomaterial_Assisted": [int(nano)],
    }
)


col_info, col_btn = st.columns([3, 1])
with col_info:
    mode_icon = "🔄" if is_continuous else "🧫"
    summary = f"{mode_icon} **{process_mode}** · {reactor} · {substrate} · {organism} · {Temp_C} °C · pH {pH:.1f}"
    summary += f" · HRT {HRT_h:.0f} h · OLR {OLR:.1f} g/L·d" if is_continuous else f" · {Time_h:.0f} h"
    st.markdown(summary)
with col_btn:
    run = st.button("Run Prediction", type="primary", use_container_width=True)


if run:
    with st.spinner("Running models..."):
        results = predict_all(hydrogen_model, vfa_model, input_df, model_meta)

    h2 = results["hydrogen_yield"]
    vfas = results["vfas"]
    dom = results["dominant_vfa"]
    eff = results["efficiency_score"]
    path = results["pathway_summary"]

    st.markdown("### Prediction Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("H2 Yield (mL/g)", f"{h2:.3f}")
    c2.metric("Dominant VFA", dom)
    c3.metric("Process Efficiency", f"{eff:.1f}%")
    c4.metric("Fermentation Mode", process_mode)

    col_pie, col_bar = st.columns(2)
    with col_pie:
        st.plotly_chart(vfa_pie(vfas), use_container_width=True)
    with col_bar:
        st.plotly_chart(vfa_bar(vfas), use_container_width=True)

    st.markdown("### Biological Pathway Analysis")
    if dom == "Acetate":
        st.success(f"**{path}**\n\nHigh acetate is associated with higher H2 yields. Conditions look favorable.")
    elif dom == "Butyrate":
        st.info(f"**{path}**\n\nButyrate pathway can be productive, but a pH near 5.5-6.0 may shift metabolism toward acetate.")
    elif dom == "Propionate":
        st.warning(f"**{path}**\n\nPropionate formation can consume H2. Check inoculum pre-treatment and substrate loading.")
    else:
        st.warning(f"**{path}**\n\nLactic acid fermentation can suggest inhibition or washout. Confirm pH and inoculum preparation.")

    if not is_continuous:
        st.caption("HRT and OLR are not applicable for batch/fed-batch systems and were excluded from this prediction.")

    with st.expander("View Full Input Parameters"):
        display = input_df.T.rename(columns={0: "Value"})
        display["Value"] = display["Value"].apply(lambda x: "N/A" if isinstance(x, float) and np.isnan(x) else x)
        st.dataframe(display, use_container_width=True)
else:
    st.info("Adjust fermentation parameters in the sidebar, then click **Run Prediction**.")
    st.markdown("### Parameter Guide")
    guide = {
        "Batch": [
            ("Time", "Total incubation time is the key time parameter for batch systems."),
            ("pH", "Dark fermentation often performs well around pH 5.5-7.0."),
            ("Temperature", "Mesophilic: 30-40 °C. Thermophilic: 50-60 °C."),
        ],
        "Continuous": [
            ("HRT", "Short HRT can cause washout; long HRT can reduce productivity."),
            ("OLR", "Higher OLR can increase productivity but may trigger inhibition."),
            ("pH", "Continuous systems benefit from active pH control."),
        ],
        "Fed-Batch": [
            ("Time", "Use total run duration including feeding cycles."),
            ("pH", "Monitor pH after substrate additions."),
        ],
    }
    for param, desc in guide.get(process_mode, guide["Batch"]):
        st.markdown(f"- **{param}:** {desc}")
