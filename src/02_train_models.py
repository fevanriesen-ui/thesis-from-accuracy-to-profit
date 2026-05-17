"""
02_train_models.py
═══════════════════════════════════════════════════════════════════════════════
Model Training — MSc Thesis: From Accuracy to Profit
F.E. van Riesen | Tilburg University | Data Science & Society | 2026
═══════════════════════════════════════════════════════════════════════════════

PURPOSE
-------
Trains two XGBoost models on identical data using different loss functions:

  Model 1 — Baseline (MSE):
    Symmetric quadratic loss (reg:squarederror). Under MSE, the optimal
    point forecast equals the conditional mean of the predictive distribution
    (Gneiting, 2011). Serves as the benchmark for all comparisons.

  Model 2 — Cost-Sensitive (Asymmetric Imbalance Price Loss):
    Custom loss function in which each forecast error is weighted by the
    realized Dutch imbalance settlement price for that PTU interval.
    Over-prediction errors (BRP is long) are weighted by |price_surplus|.
    Under-prediction errors (BRP is short) are weighted by |price_shortage|.
    Settlement prices enter ONLY the loss function — never the feature matrix.

SINGLE MODEL CLASS RATIONALE
------------------------------
Both models use the identical XGBoost architecture and feature set.
Holding architecture constant ensures that any performance difference on the
test set is attributable solely to the loss function, not to differences in
model capacity, feature engineering, or hyperparameter choices.
XGBoost is selected for three reasons:
  1. Native support for custom gradient-based loss functions via xgb.train()
  2. Strong benchmark performance on structured tabular time-series data
  3. Established use in energy forecasting literature
     (Zhang et al., 2021; Wu et al., 2021; Barja-Martínez et al., 2025)

ASYMMETRIC LOSS FUNCTION
-------------------------
For each PTU interval t, given forecast ŷ_t and actual load y_t:

  error_t = ŷ_t - y_t

  weight_t = |price_surplus_t|   if error_t > 0  (over-prediction: BRP is long)
           = |price_shortage_t|  if error_t ≤ 0  (under-prediction: BRP is short)

  loss_t   = weight_t × error_t²

  gradient_t = weight_t × 2 × error_t
  hessian_t  = weight_t × 2

Absolute values are taken because imbalance prices can be negative in some
intervals. The quadratic (L2) form is used to ensure analytically tractable
gradients compatible with XGBoost's boosting algorithm, following Wu et al.
(2021) and Zhang et al. (2021).

In single-pricing intervals (price_shortage == price_surplus, 84.9% of PTUs),
the loss is symmetric but still price-weighted — heavier in high-price periods.
Full asymmetry applies in dual-pricing intervals (15.1% of PTUs).

DATA LEAKAGE PREVENTION
------------------------
  price_shortage and price_surplus are NEVER included in FEATURE_COLS.
  Prices enter only via the make_asymmetric_loss() closure at training time.
  The test set (2025) is never seen during training or hyperparameter tuning.
  Hyperparameters are tuned on the validation set (2024) only.
  Early stopping monitors validation MSE — not the custom loss — to avoid
  overfitting on the price-weighted objective.

INPUTS
------
  dataset.parquet        — output of 01_build_dataset.py

OUTPUTS
-------
  model_baseline.json    — trained MSE model (XGBoost booster, JSON format)
  model_cost.json        — trained cost-sensitive model (XGBoost booster)
  predictions.parquet    — predictions from both models on validation + test sets
                           columns: timestamp, split, load_mwh, price_shortage,
                                    price_surplus, pred_baseline, pred_cost

VALIDATION RESULTS (reference, from last run)
----------------------------------------------
  pred_baseline : MAE=152.4 MWh | RMSE=361.8 MWh | Bias=-128.19 MWh
  pred_cost     : MAE=155.1 MWh | RMSE=367.2 MWh | Bias=-131.62 MWh

  Both models stopped at iteration 268 (early stopping).
  Cost-sensitive model shows slightly higher statistical error — consistent
  with the expected accuracy–cost trade-off (addressed in 03_evaluate.py).

DEPENDENCIES
------------
  pip install pandas pyarrow numpy xgboost scikit-learn
  brew install libomp   (macOS only — required by XGBoost)

USAGE
-----
  python 02_train_models.py
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "processed", "dataset.parquet")
OUT_BASELINE = os.path.join(PROJECT_ROOT, "models", "model_baseline.json")
OUT_COST     = os.path.join(PROJECT_ROOT, "models", "model_cost.json")
OUT_PREDS    = os.path.join(PROJECT_ROOT, "outputs", "predictions.parquet")

# Feature matrix passed to XGBoost.
# price_shortage and price_surplus are deliberately excluded — they are used
# only in the asymmetric loss function, never as predictive features.
FEATURE_COLS = [
    "load_lag_1",       # Load 15 minutes prior (short-term autocorrelation)
    "load_lag_96",      # Load 24 hours prior (daily seasonality)
    "load_lag_672",     # Load 7 days prior (weekly seasonality)
    "temperature_c",    # Air temperature, De Bilt [°C]
    "wind_speed_ms",    # Wind speed, De Bilt [m/s]
    "hour",             # Hour of day (0–23)
    "minute",           # Minute of hour (0, 15, 30, 45)
    "day_of_week",      # Day of week (0=Monday, 6=Sunday)
    "month",            # Month of year (1–12)
    "is_weekend",       # Binary weekend flag
    "is_holiday",       # Binary Dutch public holiday flag
]

TARGET_COL = "load_mwh"  # Realized net electricity infeed [MWh per PTU]

# XGBoost hyperparameters.
# These defaults were selected based on common practice in energy forecasting.
# Systematic tuning via grid/random search on the 2024 validation set is
# recommended before final test set evaluation (see thesis methodology).
PARAMS = {
    "n_estimators":     1000,   # Maximum boosting rounds (early stopping applies)
    "learning_rate":    0.05,   # Shrinkage — lower values require more rounds
    "max_depth":        6,      # Tree depth — controls model complexity
    "subsample":        0.8,    # Row subsampling per tree (regularisation)
    "colsample_bytree": 0.8,    # Feature subsampling per tree (regularisation)
    "min_child_weight": 10,     # Minimum sum of instance weight in leaf
    "random_state":     42,     # Reproducibility seed
    "n_jobs":           -1,     # Use all available CPU cores
    "verbosity":        0,      # Suppress XGBoost console output
}

# Early stopping: halt if validation loss does not improve for N rounds.
# Prevents overfitting and determines the optimal number of boosting rounds.
EARLY_STOPPING_ROUNDS = 50

# ---------------------------------------------------------------------------
# 1. Load and split data
# ---------------------------------------------------------------------------

def load_splits(data_path: str):
    """
    Loads dataset.parquet and returns train, validation, and test splits.

    Each split is returned as a tuple of:
      X           : feature matrix (np.float32 array, shape [n, len(FEATURE_COLS)])
      y           : target vector (np.float32 array, shape [n])
      p_short     : price_shortage vector (np.float32) — for loss function only
      p_surplus   : price_surplus vector  (np.float32) — for loss function only
      timestamps  : timestamp array — for output alignment in predictions.parquet

    Data leakage check:
      Prices are extracted separately from X to make their exclusion from the
      feature matrix explicit and verifiable. They are passed to the loss
      function as a separate argument, never to model.fit() as features.
    """
    df = pd.read_parquet(data_path)

    train = df[df["split"] == "train"].copy()
    val   = df[df["split"] == "validation"].copy()
    test  = df[df["split"] == "test"].copy()

    print(f"  Train      : {len(train):,} rows")
    print(f"  Validation : {len(val):,} rows")
    print(f"  Test       : {len(test):,} rows")

    def split_xy(subset):
        X          = subset[FEATURE_COLS].values.astype(np.float32)
        y          = subset[TARGET_COL].values.astype(np.float32)
        p_short    = subset["price_shortage"].values.astype(np.float32)
        p_surplus  = subset["price_surplus"].values.astype(np.float32)
        timestamps = subset["timestamp"].values
        return X, y, p_short, p_surplus, timestamps

    return split_xy(train), split_xy(val), split_xy(test)


# ---------------------------------------------------------------------------
# 2. Custom asymmetric loss function
# ---------------------------------------------------------------------------

def make_asymmetric_loss(price_shortage: np.ndarray,
                         price_surplus:  np.ndarray):
    """
    Constructs and returns an XGBoost-compatible custom objective function
    that weights forecast errors by realized Dutch imbalance settlement prices.

    Parameters
    ----------
    price_shortage : np.ndarray
        Per-PTU imbalance price for short positions [EUR/MWh], training set.
    price_surplus : np.ndarray
        Per-PTU imbalance price for long positions [EUR/MWh], training set.

    Returns
    -------
    asymmetric_obj : callable
        A closure compatible with xgb.train(obj=...). Takes (y_pred, dtrain)
        and returns (gradient, hessian) arrays of shape [n_train].

    Implementation notes
    --------------------
    Prices are embedded in a closure so the loss function always operates on
    training-set prices only. Validation prices are never accessible from
    inside the objective — no data leakage is possible through this path.

    The absolute value of prices is used because Dutch imbalance prices can
    be negative in certain intervals. The loss weight is the economic cost
    magnitude, regardless of price sign.
    """
    abs_shortage = np.abs(price_shortage)
    abs_surplus  = np.abs(price_surplus)

    def asymmetric_obj(y_pred: np.ndarray, dtrain: xgb.DMatrix):
        y_true = dtrain.get_label()
        error  = y_pred - y_true

        # Weight by settlement price in the direction of the error:
        # Over-prediction  (error > 0): BRP is long  → cost = |price_surplus|
        # Under-prediction (error ≤ 0): BRP is short → cost = |price_shortage|
        weights = np.where(error > 0, abs_surplus, abs_shortage)

        grad = weights * 2.0 * error
        hess = weights * 2.0 * np.ones_like(error)

        return grad, hess

    return asymmetric_obj


# ---------------------------------------------------------------------------
# 3. Train baseline model (MSE)
# ---------------------------------------------------------------------------

def train_baseline(X_train, y_train, X_val, y_val):
    """
    Trains an XGBoost model using standard symmetric MSE loss.

    This is the benchmark model. Under MSE, the optimal point forecast equals
    the conditional mean of the predictive distribution (Gneiting, 2011).
    Any performance difference vs the cost-sensitive model on the test set
    is attributable to the loss function change alone.

    Early stopping monitors validation RMSE to prevent overfitting.
    The best_iteration attribute records the optimal number of boosting rounds
    and is logged for reproducibility.
    """
    print("\n  Training baseline model (MSE)...")

    model = xgb.XGBRegressor(
        **PARAMS,
        objective="reg:squarederror",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print(f"  Best iteration: {model.best_iteration}")
    return model


# ---------------------------------------------------------------------------
# 4. Train cost-sensitive model (asymmetric imbalance price loss)
# ---------------------------------------------------------------------------

def train_cost_sensitive(X_train, y_train,
                         X_val,   y_val,
                         p_short_train, p_surplus_train):
    """
    Trains an XGBoost model using the asymmetric imbalance-price-weighted loss.

    Uses xgb.train() with a custom objective rather than XGBRegressor, because
    XGBRegressor does not support custom gradient functions. The two interfaces
    use the same underlying boosting algorithm — results are directly comparable.

    Early stopping monitors validation RMSE (the default eval metric when no
    custom metric is specified). This is a proxy for the custom loss but
    provides a consistent stopping criterion across both models.

    Data leakage note:
      Only training-set prices (p_short_train, p_surplus_train) are accepted as
      arguments and embedded in the loss-function closure. Validation and test
      prices are never visible from inside this function; they are consumed
      downstream in predict() and the evaluation scripts to compute realised
      costs from the predictions.
    """
    print("\n  Training cost-sensitive model (asymmetric imbalance price loss)...")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    # Embed training prices in the loss function closure
    obj = make_asymmetric_loss(p_short_train, p_surplus_train)

    # Map PARAMS to xgb.train() format (different key conventions)
    params = {k: v for k, v in PARAMS.items()
              if k not in ["n_estimators", "random_state", "n_jobs", "verbosity"]}
    params["seed"]    = 42   # replaces random_state
    params["nthread"] = -1   # replaces n_jobs

    evals_result = {}
    booster = xgb.train(
        params                = params,
        dtrain                = dtrain,
        num_boost_round       = PARAMS["n_estimators"],
        obj                   = obj,
        evals                 = [(dval, "val")],
        early_stopping_rounds = EARLY_STOPPING_ROUNDS,
        evals_result          = evals_result,
        verbose_eval          = False,
        custom_metric         = None,
    )

    print(f"  Best iteration: {booster.best_iteration}")
    return booster


# ---------------------------------------------------------------------------
# 5. Generate predictions
# ---------------------------------------------------------------------------

def predict(model_baseline, booster_cost,
            X_val,  X_test,
            ts_val, ts_test,
            y_val,  y_test,
            p_short_val,  p_surplus_val,
            p_short_test, p_surplus_test):
    """
    Generates predictions from both models on validation and test sets.

    Returns a single long-format DataFrame with columns:
      timestamp      : PTU interval start
      split          : 'validation' or 'test'
      load_mwh       : realized actual load (target variable)
      price_shortage : settlement price for short positions [EUR/MWh]
      price_surplus  : settlement price for long positions [EUR/MWh]
      pred_baseline  : forecast from MSE model [MWh]
      pred_cost      : forecast from cost-sensitive model [MWh]

    This file is the primary input to 03_evaluate.py.
    Prices are included so that imbalance costs can be computed directly
    from this file without re-loading the full dataset.
    """
    pred_baseline_val  = model_baseline.predict(X_val)
    pred_baseline_test = model_baseline.predict(X_test)
    pred_cost_val      = booster_cost.predict(xgb.DMatrix(X_val))
    pred_cost_test     = booster_cost.predict(xgb.DMatrix(X_test))

    def make_df(ts, y, p_short, p_surplus, pred_base, pred_cost, split):
        return pd.DataFrame({
            "timestamp":      ts,
            "split":          split,
            "load_mwh":       y,
            "price_shortage": p_short,
            "price_surplus":  p_surplus,
            "pred_baseline":  pred_base,
            "pred_cost":      pred_cost,
        })

    return pd.concat([
        make_df(ts_val,  y_val,  p_short_val,  p_surplus_val,
                pred_baseline_val,  pred_cost_val,  "validation"),
        make_df(ts_test, y_test, p_short_test, p_surplus_test,
                pred_baseline_test, pred_cost_test, "test"),
    ], ignore_index=True)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    (X_train, y_train, p_short_train, p_surplus_train, _),  \
    (X_val,   y_val,   p_short_val,   p_surplus_val,   ts_val),  \
    (X_test,  y_test,  p_short_test,  p_surplus_test,  ts_test) = \
        load_splits(DATA_PATH)

    # Train both models
    model_baseline = train_baseline(X_train, y_train, X_val, y_val)
    booster_cost   = train_cost_sensitive(
        X_train, y_train, X_val, y_val,
        p_short_train, p_surplus_train,
    )

    # Save trained models
    model_baseline.save_model(OUT_BASELINE)
    booster_cost.save_model(OUT_COST)
    print(f"\n  Saved: {OUT_BASELINE}")
    print(f"  Saved: {OUT_COST}")

    # Generate and save predictions
    print("\nGenerating predictions...")
    preds = predict(
        model_baseline, booster_cost,
        X_val,  X_test,
        ts_val, ts_test,
        y_val,  y_test,
        p_short_val,  p_surplus_val,
        p_short_test, p_surplus_test,
    )
    preds.to_parquet(OUT_PREDS, index=False)
    print(f"  Saved: {OUT_PREDS}")

    # Sanity check on validation set
    print("\n--- Prediction Summary (Validation Set) ---")
    val = preds[preds["split"] == "validation"]
    for col in ["pred_baseline", "pred_cost"]:
        r    = val["load_mwh"] - val[col]
        mae  = r.abs().mean()
        rmse = np.sqrt((r**2).mean())
        bias = r.mean()
        print(f"  {col:20s} | MAE={mae:.1f} | RMSE={rmse:.1f} | Bias={bias:.2f}")

    print("\nDone. Run 03_evaluate.py for full results.")


if __name__ == "__main__":
    main()