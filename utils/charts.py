import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        font=dict(family="Inter, Segoe UI, sans-serif", color="#17202a"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.72)",
        margin=dict(l=20, r=20, t=58, b=35),
    )
    return fig


def vfa_pie(vfas: dict, title: str = "Predicted VFA Distribution") -> go.Figure:
    df = pd.DataFrame({"VFA": list(vfas.keys()), "Value": list(vfas.values())})
    fig = px.pie(df, names="VFA", values="Value", title=title, hole=0.46, color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(textinfo="label+percent")
    fig.update_layout(showlegend=True)
    return _layout(fig)


def vfa_bar(vfas: dict, title: str = "Predicted VFA Concentrations (g/L)") -> go.Figure:
    colors = {"Acetate": "#16856f", "Butyrate": "#2864a6", "Propionate": "#c7831a", "Lactate": "#c94f4f"}
    fig = go.Figure(
        go.Bar(
            x=list(vfas.keys()),
            y=list(vfas.values()),
            marker_color=[colors.get(key, "#95a5a6") for key in vfas.keys()],
            text=[f"{val:.3f}" for val in vfas.values()],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="VFA",
        yaxis_title="Concentration (g/L)",
        yaxis=dict(range=[0, max(vfas.values()) * 1.3 + 0.1]),
    )
    return _layout(fig)


def yield_histogram(df: pd.DataFrame, col: str = "Hydrogen_Yield") -> go.Figure:
    label = col.replace("_", " ")
    fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {label}", labels={col: label}, color_discrete_sequence=["#2864a6"])
    fig.update_layout(bargap=0.05)
    return _layout(fig)


def substrate_bar(df: pd.DataFrame) -> go.Figure:
    counts = df["Substrate"].value_counts().reset_index()
    counts.columns = ["Substrate", "Count"]
    fig = px.bar(counts, x="Substrate", y="Count", title="Experimental Runs by Substrate", color="Count", color_continuous_scale="Blues")
    fig.update_layout(xaxis_tickangle=-30)
    return _layout(fig)


def organism_bar(df: pd.DataFrame) -> go.Figure:
    counts = df["Organism"].value_counts().reset_index()
    counts.columns = ["Organism", "Count"]
    fig = px.bar(counts, x="Count", y="Organism", orientation="h", title="Experimental Runs by Organism", color="Count", color_continuous_scale="Teal")
    return _layout(fig)


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    corr = df[num_cols].corr()
    fig = px.imshow(corr, title="Pearson Correlation Matrix", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f", aspect="auto")
    return _layout(fig)


def yield_by_organism(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df,
        x="Organism",
        y="Hydrogen_Yield",
        title="Hydrogen Yield by Organism",
        color="Organism",
        labels={"Hydrogen_Yield": "H2 Yield"},
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(xaxis_tickangle=-30, showlegend=False)
    return _layout(fig)


def yield_vs_temp(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="Temp_C",
        y="Hydrogen_Yield",
        color="Substrate",
        title="Temperature vs Hydrogen Yield",
        labels={"Temp_C": "Temperature (°C)", "Hydrogen_Yield": "H2 Yield"},
        opacity=0.72,
    )
    return _layout(fig)


def yield_vs_ph(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="pH",
        y="Hydrogen_Yield",
        color="Reactor_Type",
        title="pH vs Hydrogen Yield",
        labels={"Hydrogen_Yield": "H2 Yield"},
        opacity=0.72,
    )
    return _layout(fig)


def actual_vs_predicted(y_true, y_pred, title: str = "Actual vs Predicted") -> go.Figure:
    max_val = max(max(y_true), max(y_pred)) * 1.05
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", marker=dict(color="#2864a6", opacity=0.65, size=7), name="Predictions"))
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", line=dict(color="#c94f4f", dash="dash"), name="Perfect fit"))
    fig.update_layout(title=title, xaxis_title="Actual", yaxis_title="Predicted")
    return _layout(fig)
