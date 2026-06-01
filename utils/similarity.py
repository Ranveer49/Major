import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from utils.preprocess import CATEGORICAL_COLS, FEATURE_COLS, NUMERICAL_COLS


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns for distance computation."""
    cat_dummies = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=False)
    num_part = df[NUMERICAL_COLS].copy()
    return pd.concat([num_part, cat_dummies], axis=1)


def find_similar_experiments(input_df: pd.DataFrame, dataset_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Retrieve top-k similar experiments using scaled numeric and one-hot categorical features."""
    for col in CATEGORICAL_COLS:
        if col not in input_df.columns:
            input_df[col] = dataset_df[col].mode()[0]
    for col in NUMERICAL_COLS:
        if col not in input_df.columns:
            input_df[col] = dataset_df[col].median()

    combined = pd.concat([dataset_df[FEATURE_COLS], input_df[FEATURE_COLS]], ignore_index=True)
    encoded = _encode_categoricals(combined)

    scaler = StandardScaler()
    num_idx = list(range(len(NUMERICAL_COLS)))
    encoded_arr = encoded.values.astype(float)
    encoded_arr[:, num_idx] = scaler.fit_transform(encoded_arr[:, num_idx])

    sims = cosine_similarity(encoded_arr[[-1]], encoded_arr[:-1])[0]
    result = dataset_df.copy().reset_index(drop=True)
    result["Similarity"] = (sims * 100).round(1)
    return result.nlargest(top_k, "Similarity")


def similarity_report(top_df: pd.DataFrame) -> str:
    """Return a short markdown summary of the closest experiment."""
    if top_df.empty:
        return "No similar experiments found."

    best = top_df.iloc[0]
    return "\n".join(
        [
            f"**Closest match** ({best.get('Similarity', '?')}% similarity):",
            f"- Substrate: {best.get('Substrate', 'N/A')}",
            f"- Organism: {best.get('Organism', 'N/A')}",
            f"- Temperature: {best.get('Temp_C', 'N/A')} °C",
            f"- pH: {best.get('pH', 'N/A')}",
            f"- Reactor: {best.get('Reactor_Type', 'N/A')}",
            f"- Reported H2 Yield: {best.get('Hydrogen_Yield', 'N/A')} mL/g",
        ]
    )
