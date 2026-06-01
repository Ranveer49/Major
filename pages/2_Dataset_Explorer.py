import os

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.charts import correlation_heatmap, yield_histogram, yield_vs_ph, yield_vs_temp
from utils.preprocess import clean_dataframe, generate_synthetic_dataset
from utils.ui import apply_app_style, page_header


st.set_page_config(page_title="Dataset Explorer", page_icon="📊", layout="wide")
apply_app_style()
page_header(
    "Explore training data",
    "Dataset Explorer",
    "Filter fermentation records, inspect yield distributions, and download the subset you want to use for reporting.",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    path = "dataset/processed_dataset.csv"
    raw = pd.read_csv(path) if os.path.exists(path) else generate_synthetic_dataset()
    return clean_dataframe(raw)


df = load_data()
if df.empty:
    st.error("Dataset empty after cleaning. Check your CSV.")
    st.stop()


st.sidebar.header("Filters")


def _opts(col):
    return ["All"] + sorted(df[col].dropna().unique().tolist())


sel_sub = st.sidebar.selectbox("Substrate Category", _opts("Substrate_Category"))
sel_org = st.sidebar.selectbox("Organism Group", _opts("Organism_Group"))
sel_rx = st.sidebar.selectbox("Reactor Group", _opts("Reactor_Group"))

temp_min, temp_max = float(df["Temp_C"].min()), float(df["Temp_C"].max())
ph_min, ph_max = float(df["pH"].min()), float(df["pH"].max())
temp_range = st.sidebar.slider("Temperature (°C)", temp_min, temp_max, (temp_min, temp_max))
ph_range = st.sidebar.slider("pH range", ph_min, ph_max, (ph_min, ph_max), step=0.1)

mask = df["Temp_C"].between(*temp_range) & df["pH"].between(*ph_range)
if sel_sub != "All":
    mask &= df["Substrate_Category"] == sel_sub
if sel_org != "All":
    mask &= df["Organism_Group"] == sel_org
if sel_rx != "All":
    mask &= df["Reactor_Group"] == sel_rx
filtered = df[mask].reset_index(drop=True)


c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", len(filtered))
c2.metric("Substrates", filtered["Substrate_Category"].nunique())
c3.metric("Organism Groups", filtered["Organism_Group"].nunique())
c4.metric("Avg H2 Yield", f"{filtered['Hydrogen_Yield'].mean():.3f}")
c5.metric("Yield Unit", "mol H2/mol")

with st.expander("Raw Data Table", expanded=False):
    show_cols = [
        "Substrate_Category",
        "Organism_Group",
        "Reactor_Group",
        "Culture_Type",
        "Temp_C",
        "pH",
        "Time_h",
        "HRT_h",
        "OLR_gL_d",
        "Hydrogen_Yield",
    ]
    show_cols = [col for col in show_cols if col in filtered.columns]
    st.dataframe(filtered[show_cols], use_container_width=True, height=320)
    st.download_button(
        "Download CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_dataset.csv",
        mime="text/csv",
    )

st.markdown("### Distribution Charts")

left, right = st.columns(2)
with left:
    st.plotly_chart(yield_histogram(filtered), use_container_width=True)
with right:
    counts = filtered["Substrate_Category"].value_counts().reset_index()
    counts.columns = ["Substrate_Category", "Count"]
    fig = px.bar(
        counts,
        x="Substrate_Category",
        y="Count",
        title="Runs by Substrate Category",
        color="Count",
        color_continuous_scale="Teal",
    )
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    counts = filtered["Organism_Group"].value_counts().reset_index()
    counts.columns = ["Organism_Group", "Count"]
    fig = px.bar(
        counts,
        x="Count",
        y="Organism_Group",
        orientation="h",
        title="Runs by Organism Group",
        color="Count",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig_box = px.box(
        filtered,
        x="Organism_Group",
        y="Hydrogen_Yield",
        color="Organism_Group",
        title="Yield by Organism Group",
        labels={"Hydrogen_Yield": "H2 Yield (mol/mol)"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_box.update_layout(xaxis_tickangle=-30, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

st.markdown("### Operating Conditions vs Yield")
left, right = st.columns(2)
with left:
    st.plotly_chart(yield_vs_temp(filtered), use_container_width=True)
with right:
    st.plotly_chart(yield_vs_ph(filtered), use_container_width=True)

st.markdown("### Correlation Heatmap")
st.plotly_chart(correlation_heatmap(filtered), use_container_width=True)
