"""
utils/preprocess.py
-------------------
Feature definitions, preprocessing pipeline, data cleaning, and
organism/reactor grouping logic.

Key design decisions vs the original:
  - CATEGORICAL_COLS now use grouped columns (Organism_Group, Reactor_Group)
    instead of raw high-cardinality strings → better generalisation.
  - Engineered features added: TempxpH interaction, Low_pH flag, High_Temp
    flag, Waste_Substrate flag.
  - Hydrogen_Yield is log-transformed at training time and inverse-transformed
    at prediction time (stored as LOG_TRANSFORM = True).
  - Numeric imputer uses 'median'; blank strings in Time_h etc. are converted
    to NaN in clean_dataframe() before reaching the pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

CATEGORICAL_COLS = [
    'Substrate_Category',
    'Organism_Group',    # grouped from raw Organism (see group_organism)
    'Reactor_Group',     # grouped from raw Reactor_Type (see group_reactor)
    'Culture_Type',
]

NUMERICAL_COLS = [
    'Temp_C',
    'pH',
    'Time_h',
    'HRT_h',
    'OLR_gL_d',
    'Continuous_System',
    'Thermophilic',
    'Electrofermentation',
    'Nanomaterial_Assisted',
    # engineered
    'TempxpH',
    'Low_pH',
    'High_Temp',
    'Waste_Substrate',
]

FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS

TARGET_H2 = 'Hydrogen_Yield'

# VFAs are stored in the Metabolites text column; no separate numeric VFA cols
# exist in the real dataset. The VFA model uses the dominant metabolite flag.
VFA_TARGETS = ['Acetate', 'Butyrate', 'Propionate', 'Lactate']

# Use log1p transform on the target (critical for skewed bioprocess data)
LOG_TRANSFORM = True

# Only rows with these yield units are comparable — others have incompatible
# scales (cm³/gVS vs mol/mol) that inject noise.
MOLAR_UNITS = [
    'molH2_molGlucose', 'molH2_molSugar', 'molH2_molHexose',
    'molH2_molSubstrate', 'molH2_molCellulose', 'molH2_molSucrose',
    'molH2_molDextrose', 'molH2_molGlycerol', 'molH2_molFructose',
    'molH2_molGalactose', 'molH2_molArabinose', 'molH2_molXylose',
    'molH2_molLactose', 'molH2_molMaltose', 'mmol_H2_molGlycerol',
]

YIELD_CAP = 10.0   # mol/mol — biological maximum ~4 for glucose; cap removes
                   # likely unit-conversion errors (e.g. the 324 mmol/mol row)

# ---------------------------------------------------------------------------
# Biological grouping maps
# ---------------------------------------------------------------------------

ORGANISM_MAP = {
    'clostridium':          'Clostridium',
    'enterobacter':         'Enterobacter',
    'escherichia':          'E_coli',
    'klebsiella':           'Klebsiella',
    'thermoanaerobacterium':'Thermophile',
    'caldicellulosiruptor': 'Thermophile',
    'thermotoga':           'Thermophile',
    'mixed':                'MixedCulture',
    'anaerobic_sludge':     'MixedCulture',
    'mpc':                  'MixedCulture',
    'rsc':                  'MixedCulture',
    'citrobacter':          'Citrobacter',
    'bacillus':             'Bacillus',
}

WASTE_CATEGORIES = {'Food_Waste', 'Industrial_Waste', 'VFA_Effluent', 'Dairy_Waste'}


def group_organism(name: str) -> str:
    """Map a raw organism string to a coarse biological group."""
    if pd.isna(name):
        return 'Other'
    n = str(name).lower().replace(' ', '_')
    for key, group in ORGANISM_MAP.items():
        if key in n:
            return group
    return 'Other'


def group_reactor(r: str) -> str:
    """Map a raw reactor string to a coarse reactor class."""
    if pd.isna(r):
        return 'Batch'
    r = str(r).lower()
    if 'batch' in r:
        return 'Batch'
    if 'cstr' in r or 'continuous' in r:
        return 'CSTR'
    if 'uasb' in r:
        return 'UASB'
    if 'immob' in r or 'fixed' in r or 'bed' in r:
        return 'Immobilized'
    return 'Other'


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for the real CSV dataset.

    Steps
    -----
    1. Filter to molar-unit rows only.
    2. Cap Hydrogen_Yield at YIELD_CAP to remove likely errors.
    3. Fix blank strings in Time_h (and any other numeric column).
    4. Add grouped categorical columns.
    5. Add engineered numerical features.
    6. Drop rows without a valid Hydrogen_Yield.

    Returns a clean DataFrame ready for feature extraction.
    """
    df = df.copy()

    # 1. Unit filter
    if 'Yield_Unit' in df.columns:
        df = df[df['Yield_Unit'].isin(MOLAR_UNITS)]

    # 2. Cap outlier yields
    if TARGET_H2 in df.columns:
        df = df[df[TARGET_H2] <= YIELD_CAP]

    # 3. Fix numeric columns that may contain blank strings
    numeric_repair = ['Temp_C', 'pH', 'Time_h', 'HRT_h', 'OLR_gL_d',
                      TARGET_H2, 'HPR', 'HPR_mL_L_h_standardized']
    for col in numeric_repair:
        if col in df.columns:
            df[col] = df[col].replace(['', ' ', 'NA', 'NaN',
                                       'Not_Applicable', 'Steady_State'], pd.NA)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Grouped categoricals
    df['Organism_Group'] = df['Organism'].apply(group_organism)
    df['Reactor_Group']  = df['Reactor_Type'].apply(group_reactor)

    # Fill missing Culture_Type
    if 'Culture_Type' not in df.columns:
        df['Culture_Type'] = 'MixedCulture'
    df['Culture_Type'] = df['Culture_Type'].fillna('MixedCulture')

    # 5. Engineered features
    df['TempxpH']        = df['Temp_C'] * df['pH']
    df['Low_pH']         = (df['pH'] < 5.5).astype(float)
    df['High_Temp']      = (df['Temp_C'] > 55).astype(float)
    df['Waste_Substrate'] = df['Substrate_Category'].isin(WASTE_CATEGORIES).astype(float)

    # 6. Drop missing target
    df = df.dropna(subset=[TARGET_H2])

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def create_preprocessor():
    """
    Build a ColumnTransformer for FEATURE_COLS.

    Categorical → impute(most_frequent) → OneHotEncode
    Numerical   → impute(median)        → StandardScale
    """
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    return ColumnTransformer(
        transformers=[
            ('cat', cat_pipe, CATEGORICAL_COLS),
            ('num', num_pipe, NUMERICAL_COLS),
        ],
        remainder='drop'
    )


# ---------------------------------------------------------------------------
# Input validation for prediction UI
# ---------------------------------------------------------------------------

def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a prediction input DataFrame has all FEATURE_COLS.
    Applies the same grouping and engineering as clean_dataframe.
    """
    df = df.copy()

    # Derive grouped columns if raw columns are present
    if 'Organism' in df.columns and 'Organism_Group' not in df.columns:
        df['Organism_Group'] = df['Organism'].apply(group_organism)
    if 'Reactor_Type' in df.columns and 'Reactor_Group' not in df.columns:
        df['Reactor_Group'] = df['Reactor_Type'].apply(group_reactor)

    # Engineered features
    if 'Temp_C' in df.columns and 'pH' in df.columns:
        df['TempxpH'] = df['Temp_C'] * df['pH']
        df['Low_pH']  = (df['pH'] < 5.5).astype(float)
        df['High_Temp'] = (df['Temp_C'] > 55).astype(float)
    if 'Substrate_Category' in df.columns:
        df['Waste_Substrate'] = df['Substrate_Category'].isin(WASTE_CATEGORIES).astype(float)

    # Fill missing columns with sensible defaults
    defaults = {
        'Substrate_Category': 'Simple_Sugar',
        'Organism_Group':     'MixedCulture',
        'Reactor_Group':      'Batch',
        'Culture_Type':       'MixedCulture',
        'Temp_C':    37.0,
        'pH':         6.5,
        'Time_h':    24.0,
        'HRT_h':     np.nan,
        'OLR_gL_d':  np.nan,
        'Continuous_System':     0,
        'Thermophilic':          0,
        'Electrofermentation':   0,
        'Nanomaterial_Assisted': 0,
        'TempxpH':    37.0 * 6.5,
        'Low_pH':     0,
        'High_Temp':  0,
        'Waste_Substrate': 0,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    return df[FEATURE_COLS]


# ---------------------------------------------------------------------------
# Synthetic dataset (for testing without real data)
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
    """Biologically-informed synthetic dataset matching the real CSV structure."""
    rng = np.random.default_rng(random_state)

    substrate_categories = ['Simple_Sugar', 'Disaccharide', 'Lignocellulose',
                             'Food_Waste', 'VFA_Effluent', 'Industrial_Waste']
    organisms    = ['Clostridium', 'Enterobacter', 'E_coli',
                    'Klebsiella', 'MixedCulture', 'Thermophile']
    reactors     = ['Batch', 'CSTR', 'UASB', 'Immobilized']
    cultures     = ['PureCulture', 'MixedCulture', 'CoCulture']
    yield_units  = ['molH2_molGlucose'] * n_samples   # uniform for clean data

    sub_cat   = rng.choice(substrate_categories, n_samples)
    org_group = rng.choice(organisms, n_samples)
    rx_group  = rng.choice(reactors, n_samples)
    cult      = rng.choice(cultures, n_samples)

    temp = rng.uniform(25, 70, n_samples)
    ph   = rng.uniform(4.5, 8.0, n_samples)
    time_h = np.where(rx_group == 'Batch', rng.uniform(6, 120, n_samples), np.nan)
    hrt_h  = np.where(rx_group != 'Batch', rng.uniform(6, 96, n_samples), np.nan)
    olr    = np.where(rx_group != 'Batch', rng.uniform(2, 30, n_samples), np.nan)
    continuous   = (rx_group != 'Batch').astype(int)
    thermophilic = (temp > 50).astype(int)
    electro = rng.integers(0, 2, n_samples)
    nano    = rng.integers(0, 2, n_samples)

    # Yield simulation with biological signal
    h2 = (
        1.8
        + 1.2 * np.exp(-0.5 * ((ph - 6.5) / 0.8) ** 2)
        - 0.003 * (temp - 40) ** 2
        + 0.3 * thermophilic
        + 0.2 * electro
        + 0.15 * nano
        - 0.4 * np.isin(sub_cat, ['Food_Waste', 'Industrial_Waste']).astype(float)
        + rng.normal(0, 0.4, n_samples)
    )
    hydrogen_yield = np.clip(h2, 0.05, 9.0)

    waste = np.isin(sub_cat, ['Food_Waste', 'VFA_Effluent', 'Industrial_Waste']).astype(int)

    return pd.DataFrame({
        'Organism':           org_group,
        'Substrate':          sub_cat,
        'Substrate_Category': sub_cat,
        'Temp_C':             temp.round(1),
        'pH':                 ph.round(2),
        'Time_h':             [round(v, 1) if not np.isnan(v) else np.nan for v in time_h],
        'Reactor_Type':       rx_group,
        'Reactor_Group':      rx_group,
        'Culture_Type':       cult,
        'Process_Mode':       np.where(continuous, 'Continuous', 'Batch'),
        'Continuous_System':  continuous,
        'Thermophilic':       thermophilic,
        'Electrofermentation':electro,
        'Nanomaterial_Assisted': nano,
        'HRT_h':              [round(v, 1) if not np.isnan(v) else np.nan for v in hrt_h],
        'OLR_gL_d':           [round(v, 2) if not np.isnan(v) else np.nan for v in olr],
        'Hydrogen_Yield':     hydrogen_yield.round(3),
        'Yield_Unit':         yield_units,
        'Organism_Group':     org_group,
        'TempxpH':            (temp * ph).round(2),
        'Low_pH':             (ph < 5.5).astype(int),
        'High_Temp':          (temp > 55).astype(int),
        'Waste_Substrate':    waste,
        # Synthetic VFA proxies
        'Acetate':    np.clip(hydrogen_yield * 0.4 + rng.normal(0, 0.3, n_samples), 0, 5).round(3),
        'Butyrate':   np.clip(hydrogen_yield * 0.35 + rng.normal(0, 0.3, n_samples), 0, 5).round(3),
        'Propionate': np.clip(rng.uniform(0.1, 2.0, n_samples), 0, 3).round(3),
        'Lactate':    np.clip(rng.uniform(0.0, 1.5, n_samples), 0, 3).round(3),
    })
