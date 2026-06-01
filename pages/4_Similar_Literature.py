import os

import pandas as pd
import streamlit as st

from utils.preprocess import generate_synthetic_dataset
from utils.similarity import find_similar_experiments, similarity_report
from utils.ui import apply_app_style, page_header


st.set_page_config(page_title="Similar Literature", page_icon="📚", layout="wide")
apply_app_style()
page_header(
    "Compare against prior runs",
    "Similar Literature Finder",
    "Find the closest matching experiments in the dataset so predictions can be interpreted beside comparable fermentation conditions.",
)


@st.cache_data
def load_data():
    path = "dataset/processed_dataset.csv"
    return pd.read_csv(path) if os.path.exists(path) else generate_synthetic_dataset()


df = load_data()

st.markdown("### Query Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    substrate = st.selectbox("Substrate", ["Glucose", "Sucrose", "Food Waste", "Glycerol", "Starch", "Cellulose", "Lactose", "Whey"])
    organism = st.selectbox(
        "Organism",
        [
            "Mixed Culture",
            "Clostridium butyricum",
            "Clostridium pasteurianum",
            "Enterobacter aerogenes",
            "Thermoanaerobacterium",
            "Anaerobic sludge",
        ],
    )
with col2:
    temp = st.slider("Temperature (°C)", 20, 80, 37)
    ph = st.slider("pH", 4.0, 9.0, 6.5, step=0.1)
with col3:
    reactor = st.selectbox("Reactor Type", ["Batch", "CSTR", "UASB", "AnSBR"])
    top_k = st.slider("Number of results", 3, 15, 5)

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

query_df = pd.DataFrame(
    {
        "Substrate": [substrate],
        "Substrate_Category": [substrate_map.get(substrate, "Other")],
        "Organism": [organism],
        "Reactor_Type": [reactor],
        "Culture_Type": ["MixedCulture"],
        "Process_Mode": ["Batch"],
        "Temp_C": [float(temp)],
        "pH": [float(ph)],
        "Time_h": [24.0],
        "HRT_h": [24.0],
        "OLR_gL_d": [10.0],
        "Continuous_System": [0],
        "Thermophilic": [int(temp > 50)],
        "Electrofermentation": [0],
        "Nanomaterial_Assisted": [0],
    }
)

if st.button("Find Similar Experiments", type="primary"):
    with st.spinner("Computing similarity..."):
        results = find_similar_experiments(query_df, df, top_k=top_k)

    st.markdown("### Top Matches")
    st.markdown(similarity_report(results))
    display_cols = [
        col
        for col in [
            "Similarity",
            "Substrate",
            "Organism",
            "Reactor_Type",
            "Temp_C",
            "pH",
            "Time_h",
            "Hydrogen_Yield",
            "Acetate",
            "Butyrate",
            "Propionate",
            "Lactate",
        ]
        if col in results.columns
    ]
    st.dataframe(results[display_cols].reset_index(drop=True), use_container_width=True)

    yields = results["Hydrogen_Yield"].dropna()
    if not yields.empty:
        st.info(f"H2 yield range in top {top_k} matches: **{yields.min():.3f}** to **{yields.max():.3f}** mL H2/g")
else:
    st.info("Set query parameters above and click **Find Similar Experiments**.")
