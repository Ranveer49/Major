import streamlit as st


def apply_app_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bio-bg: #f6f8fb;
            --bio-panel: #ffffff;
            --bio-ink: #17202a;
            --bio-muted: #5d6d7e;
            --bio-green: #16856f;
            --bio-blue: #2864a6;
            --bio-gold: #c7831a;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(22, 133, 111, 0.10), transparent 28rem),
                linear-gradient(180deg, #f7fbf9 0%, var(--bio-bg) 42%, #ffffff 100%);
            color: var(--bio-ink);
        }

        [data-testid="stSidebar"] {
            background: #edf5f2;
            border-right: 1px solid rgba(22, 133, 111, 0.16);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label {
            color: var(--bio-ink);
        }

        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--bio-ink);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1220px;
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(40, 100, 166, 0.14);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            box-shadow: 0 10px 28px rgba(23, 32, 42, 0.05);
        }

        [data-testid="stMetricLabel"] p {
            color: var(--bio-muted);
            font-size: 0.86rem;
        }

        [data-testid="stMetricValue"] {
            color: var(--bio-green);
            font-weight: 750;
        }

        div.stButton > button,
        div.stDownloadButton > button {
            border-radius: 7px;
            border: 1px solid rgba(22, 133, 111, 0.28);
            font-weight: 650;
        }

        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--bio-green), var(--bio-blue));
            border: 0;
        }

        .bio-hero {
            padding: 1.2rem 0 0.4rem;
        }

        .bio-eyebrow {
            color: var(--bio-green);
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.78rem;
            margin-bottom: 0.4rem;
        }

        .bio-lede {
            color: var(--bio-muted);
            font-size: 1.08rem;
            line-height: 1.65;
            max-width: 820px;
        }

        .bio-panel {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(40, 100, 166, 0.13);
            border-radius: 8px;
            padding: 1rem 1.05rem;
            box-shadow: 0 10px 28px rgba(23, 32, 42, 0.045);
        }

        .bio-panel strong {
            color: var(--bio-ink);
        }

        .bio-small {
            color: var(--bio-muted);
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .bio-step {
            border-left: 3px solid var(--bio-green);
            padding: 0.55rem 0 0.55rem 0.85rem;
            margin-bottom: 0.45rem;
            background: rgba(255, 255, 255, 0.55);
            border-radius: 0 7px 7px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="bio-hero">
            <div class="bio-eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p class="bio-lede">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
