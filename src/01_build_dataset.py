"""
01_build_dataset.py
═══════════════════════════════════════════════════════════════════════════════
ETL Pipeline — MSc Thesis: From Accuracy to Profit
F.E. van Riesen | Tilburg University | Data Science & Society | 2026
═══════════════════════════════════════════════════════════════════════════════

PURPOSE
-------
Builds a clean, merged dataset at 15-minute (PTU) resolution from two
TenneT Transparency Portal sources and KNMI meteorological data. The output
is a single Parquet file ready for model training in 02_train_models.py.

INPUTS (place all files in the same directory as this script)
-------------------------------------------------------------
TenneT Transparency Portal (https://transparency.tennet.eu):
  - metered_injections_<YYYY>.csv      [Gemeten invoeding, 4 files: 2022-2025]
  - settlement_prices_<YYYY>.csv       [Verrekenprijzen,   4 files: 2022-2025]

KNMI (downloaded via API in download_knmi.py):
  - knmi_weather.csv                   [Station De Bilt (260), hourly, 2022-2025]

OUTPUT
------
  dataset.parquet — 139,568 rows × 19 columns, no missing values

COLUMNS
-------
  timestamp             : datetime64, UTC, 15-min PTU interval start
  load_mwh              : float, realized net electricity infeed [MWh]
  price_shortage        : float, imbalance price for short BRPs [EUR/MWh]
  price_surplus         : float, imbalance price for long BRPs [EUR/MWh]
  regulation_state      : int, -1 = single pricing, 2 = dual pricing
  regulating_condition  : str, DOWN / UP / UP_AND_DOWN / STABLE
  dual_pricing          : int, binary flag (1 = dual pricing active)
  temperature_c         : float, air temperature De Bilt [°C]
  wind_speed_ms         : float, wind speed De Bilt [m/s], abs() applied
  hour                  : int, hour of day (0-23)
  minute                : int, minute of hour (0, 15, 30, 45)
  day_of_week           : int, 0=Monday, 6=Sunday
  month                 : int, 1-12
  is_weekend            : int, binary (1 = Saturday or Sunday)
  is_holiday            : int, binary (1 = Dutch public holiday)
  load_lag_1            : float, load 1 PTU (15 min) prior [MWh]
  load_lag_96           : float, load 96 PTUs (24 hours) prior [MWh]
  load_lag_672          : float, load 672 PTUs (7 days) prior [MWh]
  split                 : str, train / validation / test

DATA SPLITS
-----------
  train      : 2022-2023  (~70,000 obs) — model training
  validation : 2024       (~35,000 obs) — hyperparameter tuning
  test       : 2025       (~35,000 obs) — held-out, used once only

PREPROCESSING DECISIONS
-----------------------
  1. DST duplicates    : Removed (October clock-back, 4 years)
                         First occurrence per timestamp retained.
  2. Negative wind     : KNMI values converted via abs() — KNMI encodes
                         calm/variable wind as negative speed.
  3. Negative load     : Retained — physically real net export
                         intervals (solar surplus, midday Mar-Oct 2024-2025).
  4. Lag NaN rows      : 672 rows dropped (first 7 days, lag_672 window).
  5. No normalisation  : XGBoost does not require feature scaling.
  6. No leakage        : All lags are strictly backward-looking (.shift()).
                         Prices never enter the feature matrix X.

DEPENDENCIES
------------
  pip install pandas pyarrow numpy holidays

USAGE
-----
  python 01_build_dataset.py

REPRODUCIBILITY
---------------
  Raw data sources are publicly available without authentication.
  This script is deterministic: same inputs always produce same output.
  GitHub repository: [add URL before submission]
═══════════════════════════════════════════════════════════════════════════════
"""

import glob
import os

import holidays
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------

# Raw CSVs (TenneT + KNMI) live in data/raw/; processed dataset is written
# to data/processed/. Both paths are resolved relative to the project root,
# which is the parent of src/ (this script's directory).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_PATH     = os.path.join(PROJECT_ROOT, "data", "processed", "dataset.parquet")

# Dutch public holidays used for the is_holiday feature
NL_HOLIDAYS = holidays.Netherlands(years=[2022, 2023, 2024, 2025])

# ---------------------------------------------------------------------------
# 1. Load & concatenate metered injections (target variable)
# ---------------------------------------------------------------------------

def load_metered_injections(base_dir: str) -> pd.DataFrame:
    """
    Loads all metered_injections_*.csv files from TenneT.

    Source  : TenneT Transparency Portal > Gemeten invoeding
    Variable: Measured Infeed [MWh per 15-min PTU]
    Notes   : Negative values (~1.85%) represent net export intervals
              caused by solar surplus; these are retained as physically real.

    Returns
    -------
    pd.DataFrame with columns [timestamp, load_mwh]
    """
    files = sorted(glob.glob(os.path.join(base_dir, "metered_injections_*.csv")))
    if not files:
        raise FileNotFoundError(
            "No metered_injections_*.csv files found in data/raw/. "
            "Download from: https://transparency.tennet.eu > Gemeten invoeding"
        )

    frames = []
    for f in files:
        df = pd.read_csv(f, sep=None, engine="python")
        df.columns = df.columns.str.strip()
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    raw["timestamp"] = pd.to_datetime(
        raw["Timeinterval Start Loc"], format="%Y-%m-%dT%H:%M"
    )

    out = raw[["timestamp", "Measured Infeed"]].copy()
    out.columns = ["timestamp", "load_mwh"]
    out = out.sort_values("timestamp").reset_index(drop=True)

    print(f"  Metered injections: {len(out):,} rows loaded.")
    return out


# ---------------------------------------------------------------------------
# 2. Load & concatenate settlement prices
# ---------------------------------------------------------------------------

def load_settlement_prices(base_dir: str) -> pd.DataFrame:
    """
    Loads all settlement_prices_*.csv files from TenneT.

    Source  : TenneT Transparency Portal > Verrekenprijzen
    Variables:
      price_shortage   : Price paid by short BRPs [EUR/MWh]
                         Used as loss weight for under-prediction errors
                         in the cost-sensitive model (02_train_models.py).
      price_surplus    : Price paid by long BRPs [EUR/MWh]
                         Used as loss weight for over-prediction errors.
      regulation_state : -1 = single pricing (DOWN only)
                          2 = dual pricing (UP_AND_DOWN)
      dual_pricing     : Binary flag derived from regulation_state.

    IMPORTANT: price_shortage and price_surplus are used ONLY in the loss
    function during training. They are never passed to XGBoost as features.

    Returns
    -------
    pd.DataFrame with columns [timestamp, price_shortage, price_surplus,
                                regulation_state, regulating_condition,
                                dual_pricing]
    """
    files = sorted(glob.glob(os.path.join(base_dir, "settlement_prices_*.csv")))
    if not files:
        raise FileNotFoundError(
            "No settlement_prices_*.csv files found in data/raw/. "
            "Download from: https://transparency.tennet.eu > Verrekenprijzen"
        )

    frames = []
    for f in files:
        df = pd.read_csv(f, sep=None, engine="python")
        df.columns = df.columns.str.strip()
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    raw["timestamp"] = pd.to_datetime(
        raw["Timeinterval Start Loc"], format="%Y-%m-%dT%H:%M"
    )

    out = raw[[
        "timestamp",
        "Price Shortage",
        "Price Surplus",
        "Regulation State",
        "Regulating Condition"
    ]].copy()

    out.columns = [
        "timestamp",
        "price_shortage",
        "price_surplus",
        "regulation_state",
        "regulating_condition"
    ]

    # Dual pricing flag: Regulation State == 2 means UP_AND_DOWN (dual pricing)
    # Single pricing:    Regulation State == -1 means DOWN only
    out["dual_pricing"] = (out["regulation_state"] == 2).astype(int)
    out = out.sort_values("timestamp").reset_index(drop=True)

    print(f"  Settlement prices: {len(out):,} rows loaded.")
    return out


# ---------------------------------------------------------------------------
# 3. Load KNMI weather data and resample to 15-minute resolution
# ---------------------------------------------------------------------------

def load_knmi_weather(base_dir: str) -> pd.DataFrame:
    """
    Loads KNMI hourly weather data and resamples to 15-minute resolution.

    Source  : KNMI Climate Data Portal (daggegevens.knmi.nl)
              Station: De Bilt (station 260) — standard reference for NL
    Download: See download_knmi.py (API call, no authentication required)

    Variables:
      temperature_c  : Air temperature [°C] (KNMI raw unit: 0.1°C, divided by 10)
      wind_speed_ms  : Wind speed [m/s]     (KNMI raw unit: 0.1 m/s, divided by 10)

    Preprocessing:
      - KNMI hour convention: 1-24, where hour 24 = midnight next day
        Handled via: hour % 24 with date adjustment
      - Negative wind speed values (3.4% of obs) reflect KNMI's encoding
        of calm or variable wind direction — converted via abs()
      - Hourly data forward-filled to 15-minute resolution to match TenneT

    Returns
    -------
    pd.DataFrame with columns [timestamp, temperature_c, wind_speed_ms]
    at 15-minute resolution
    """
    path = os.path.join(base_dir, "knmi_weather.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "knmi_weather.csv not found in data/raw/. "
            "See README.md for KNMI download instructions."
        )

    raw = pd.read_csv(path, header=None, skipinitialspace=True)
    raw.columns = ["station", "date", "hour", "wind_speed_raw", "temperature_raw"]

    # KNMI hour runs 1-24; hour 24 means midnight of the next day
    raw["hour_adj"] = raw["hour"] % 24
    raw["date_str"] = raw["date"].astype(str)
    raw["timestamp"] = (
        pd.to_datetime(raw["date_str"], format="%Y%m%d")
        + pd.to_timedelta(raw["hour_adj"], unit="h")
    )

    # Convert units: KNMI stores in tenths (0.1°C, 0.1 m/s)
    raw["temperature_c"] = raw["temperature_raw"] / 10.0
    raw["wind_speed_ms"] = raw["wind_speed_raw"]  / 10.0

    # Calm/variable wind is encoded as negative speed in KNMI data
    # Physical interpretation: near-zero wind, direction undefined
    raw["wind_speed_ms"] = raw["wind_speed_ms"].abs()

    hourly = raw[["timestamp", "temperature_c", "wind_speed_ms"]].copy()
    hourly = hourly.sort_values("timestamp").reset_index(drop=True)

    # Forward-fill from hourly to 15-minute resolution
    # Each hourly value is propagated to the 3 subsequent 15-min slots
    hourly = hourly.set_index("timestamp")
    quarterly = hourly.resample("15min").ffill()
    quarterly = quarterly.reset_index()

    print(f"  KNMI weather: {len(quarterly):,} rows after resampling to 15-min.")
    return quarterly


# ---------------------------------------------------------------------------
# 4. Fix DST duplicate timestamps
# ---------------------------------------------------------------------------

def fix_dst_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate timestamps caused by daylight saving time clock-back.

    The Dutch DST transition (last Sunday of October) causes clocks to go
    back from 03:00 to 02:00 CET, producing one hour that occurs twice.
    This creates 8 duplicate PTUs per year (02:00-02:45), or 32 over 2022-2025.
    After merging the TenneT and KNMI sources, duplicates may be multiplied;
    the resolution is to sort by timestamp and keep the first occurrence (CET).
    """
    n_before = len(df)
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="first"
    )
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} duplicate timestamps (DST clock-back).")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Add calendar features
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds time-based features derived from the timestamp.

    These capture structural load patterns without requiring external data:
      hour        : Intraday demand cycle (peak ~08:00-20:00)
      minute      : Sub-hourly position within the PTU (0, 15, 30, 45)
      day_of_week : Weekly demand pattern (weekday vs weekend)
      month       : Seasonal demand pattern
      is_weekend  : Binary weekend flag
      is_holiday  : Binary Dutch public holiday flag (holidays library)
    """
    df["hour"]        = df["timestamp"].dt.hour
    df["minute"]      = df["timestamp"].dt.minute
    df["day_of_week"] = df["timestamp"].dt.dayofweek   # 0=Monday, 6=Sunday
    df["month"]       = df["timestamp"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"]  = df["timestamp"].dt.date.apply(
        lambda d: int(d in NL_HOLIDAYS)
    )
    return df


# ---------------------------------------------------------------------------
# 6. Add lag features
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds autoregressive lag features for the target variable.

    Lag conventions (PTU = 15-minute interval):
      load_lag_1   =   1 PTU  =  15 minutes prior  (short-term autocorrelation)
      load_lag_96  =  96 PTUs =  24 hours prior     (daily seasonality)
      load_lag_672 = 672 PTUs =   7 days prior      (weekly seasonality)

    Data leakage prevention:
      All lags use .shift(n), which strictly references past observations.
      Row t only sees values from rows t-1, t-96, t-672.
      The first 672 rows are dropped in main() due to NaN lag values.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["load_lag_1"]   = df["load_mwh"].shift(1)
    df["load_lag_96"]  = df["load_mwh"].shift(96)
    df["load_lag_672"] = df["load_mwh"].shift(672)
    return df


# ---------------------------------------------------------------------------
# 7. Assign train / validation / test split
# ---------------------------------------------------------------------------

def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns a temporal split label to each observation.

    Split design (full calendar years):
      train      : 2022-2023  — model fitting (~70,000 obs)
      validation : 2024       — hyperparameter tuning (~35,000 obs)
      test       : 2025       — held-out evaluation, used ONCE (~35,000 obs)

    Rationale:
      Full calendar years are used to ensure all seasons are equally
      represented in each partition. Random k-fold cross-validation is
      inappropriate for time-series data as it causes data leakage by
      allowing future observations to inform past predictions.
      The temporal split is the methodologically correct alternative for
      sequential data with seasonal structure.
    """
    year = df["timestamp"].dt.year
    df["split"] = np.select(
        [year <= 2023, year == 2024, year == 2025],
        ["train", "validation", "test"],
        default="unknown"
    )
    return df


# ---------------------------------------------------------------------------
# 8. Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("Building dataset...")

    print("\n[1/6] Loading metered injections...")
    load = load_metered_injections(RAW_DIR)

    print("\n[2/6] Loading settlement prices...")
    prices = load_settlement_prices(RAW_DIR)

    print("\n[3/6] Loading KNMI weather data...")
    weather = load_knmi_weather(RAW_DIR)

    print("\n[4/6] Merging on timestamp...")
    df = load.merge(prices,    on="timestamp", how="inner")
    df = df.merge(weather,     on="timestamp", how="inner")
    df = fix_dst_duplicates(df)
    print(f"  Merged dataset: {len(df):,} rows.")

    print("\n[5/6] Adding calendar and lag features...")
    df = add_calendar_features(df)
    df = add_lag_features(df)

    print("\n[6/6] Assigning train/validation/test split...")
    df = assign_split(df)

    # Drop rows where lag features are NaN (first 7 days of series)
    n_before = len(df)
    df = df.dropna(subset=["load_lag_1", "load_lag_96", "load_lag_672"])
    print(f"  Dropped {n_before - len(df):,} rows with NaN lag values.")

    # Final summary
    print("\n--- Dataset Summary ---")
    print(f"  Total rows    : {len(df):,}")
    print(f"  Date range    : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Columns       : {list(df.columns)}")
    print(f"\n  Split counts:")
    print(df["split"].value_counts().to_string())
    print(f"\n  Missing values:")
    print(df.isnull().sum().to_string())

    df.to_parquet(OUT_PATH, index=False)
    print(f"\n  Saved to: {OUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
