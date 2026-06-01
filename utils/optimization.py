import itertools

import numpy as np
import pandas as pd

from utils.preprocess import validate_input


DEFAULT_TEMP_RANGE = [30, 37, 45, 55, 60]
DEFAULT_PH_RANGE = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
DEFAULT_OLR_RANGE = [5.0, 10.0, 15.0, 20.0, 25.0]


def optimize_conditions(
    hydrogen_model,
    base_input: dict,
    temp_range: list | None = None,
    ph_range: list | None = None,
    olr_range: list | None = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """Grid-search operating conditions and return top predicted H2 yields."""
    temp_range = temp_range or DEFAULT_TEMP_RANGE
    ph_range = ph_range or DEFAULT_PH_RANGE
    olr_range = olr_range or DEFAULT_OLR_RANGE

    rows = []
    for temp, ph, olr in itertools.product(temp_range, ph_range, olr_range):
        row = base_input.copy()
        row["Temp_C"] = temp
        row["pH"] = ph
        row["OLR_gL_d"] = olr
        rows.append(row)

    df = pd.DataFrame(rows)
    X = validate_input(df.copy())
    preds = hydrogen_model.predict(X)
    df["Predicted_H2_Yield"] = np.clip(preds, 0, None).round(4)
    return df.sort_values("Predicted_H2_Yield", ascending=False).head(top_k).reset_index(drop=True)


def sensitivity_analysis(hydrogen_model, base_input: dict, param: str, param_range: list) -> pd.DataFrame:
    """Vary one parameter while holding other model inputs constant."""
    rows = []
    for val in param_range:
        row = base_input.copy()
        row[param] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    X = validate_input(df.copy())
    preds = hydrogen_model.predict(X)
    return pd.DataFrame({param: param_range, "Predicted_H2_Yield": np.clip(preds, 0, None).round(4)})


def optimization_report(opt_df: pd.DataFrame) -> str:
    """Return a concise markdown summary of the best predicted conditions."""
    if opt_df.empty:
        return "Optimization did not return any results."

    best = opt_df.iloc[0]
    return "\n".join(
        [
            "### Optimal Predicted Conditions",
            f"- **Temperature**: {best.get('Temp_C', 'N/A')} °C",
            f"- **pH**: {best.get('pH', 'N/A')}",
            f"- **OLR**: {best.get('OLR_gL_d', 'N/A')} g/L·d",
            f"- **Predicted H2 Yield**: {best.get('Predicted_H2_Yield', 'N/A'):.3f} mL H2/g",
        ]
    )
