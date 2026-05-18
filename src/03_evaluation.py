"""
03_evaluate.py
═══════════════════════════════════════════════════════════════════════════════
Evaluation & Analysis — MSc Thesis: From Accuracy to Profit
F.E. van Riesen | Tilburg University | Data Science & Society | 2026
═══════════════════════════════════════════════════════════════════════════════

PURPOSE
-------
Evaluates both models on the held-out 2025 test set and produces all results
tables and figures needed for the thesis Results section.

RESEARCH QUESTIONS ADDRESSED
------------------------------
  RQ1 — Economic Performance:
    To what extent do models trained under an imbalance-price-based asymmetric
    loss differ from MSE-trained models in terms of realized imbalance costs?
    → Table 1: Imbalance cost per PTU (mean, total, reduction %)

  RQ2 — Forecast Behavior:
    How does asymmetric training affect forecast bias and directional error
    patterns across different load and imbalance price regimes?
    → Table 2: Mean bias, over/under-prediction rates by model
    → Table 3: Regime breakdown (peak/off-peak, high/normal price)
    → Figure 1: Residual distributions (histogram)
    → Figure 2: Predicted vs actual (scatter plot) — required by DSS rubric

  RQ3 — Trade-offs and Robustness:
    What trade-offs arise between imbalance costs and statistical accuracy,
    and how robust are these on the temporally held-out 2025 test year?
    → Table 4: MAE, RMSE, imbalance cost — validation vs test comparison

IMBALANCE COST CALCULATION
----------------------------
Dutch settlement rules applied per PTU interval t:

  error_t = forecast_t - actual_t

  If error_t > 0 (over-prediction, BRP is long):
    cost_t = error_t × price_surplus_t / 1000
    [price in EUR/MWh, error in MWh, divide by 1000 for MWh → EUR scaling]

  If error_t ≤ 0 (under-prediction, BRP is short):
    cost_t = |error_t| × price_shortage_t / 1000

  Note: prices can be negative (rewarding the correct imbalance direction).
  Negative costs represent revenue, not expense. This is retained.

INPUTS
------
  predictions.parquet   — output of 02_train_models.py

OUTPUTS
-------
  results_summary.csv   — Tables 1-4 combined
  fig1_residuals.png    — Residual distribution histograms (RQ2)
  fig2_pred_vs_actual.png — Predicted vs actual scatter (DSS rubric requirement)
  fig3_regime_bias.png  — Bias breakdown by regime (RQ2)
  fig4_cost_over_time.png — Cumulative imbalance cost over test year (RQ1)

DEPENDENCIES
------------
  pip install pandas pyarrow numpy matplotlib

USAGE
-----
  python 03_evaluate.py
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDS_PATH   = os.path.join(PROJECT_ROOT, "outputs", "predictions.parquet")
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "processed", "dataset.parquet")
OUT_CSV      = os.path.join(PROJECT_ROOT, "outputs", "results_summary.csv")

# Peak hours definition (standard in Dutch energy market literature)
PEAK_HOURS = list(range(8, 21))   # 08:00–20:00 CET

# High imbalance price threshold: top quartile of |price_shortage| in test set
# Computed dynamically from the data — not hardcoded
HIGH_PRICE_QUANTILE = 0.75

# Figure output paths
FIG_DIR          = os.path.join(PROJECT_ROOT, "outputs", "figures")
FIG_RESIDUALS    = os.path.join(FIG_DIR, "fig1_residuals.png")
FIG_PRED_ACTUAL  = os.path.join(FIG_DIR, "fig2_pred_vs_actual.png")
FIG_REGIME_BIAS  = os.path.join(FIG_DIR, "fig3_regime_bias.png")
FIG_CUMCOST      = os.path.join(FIG_DIR, "fig4_cost_over_time.png")
FIG_CUMCOST_REGIME = os.path.join(FIG_DIR, "fig5_cost_by_regime.png")
FIG_PRICE_DIST   = os.path.join(FIG_DIR, "fig6_price_distributions.png")

# ---------------------------------------------------------------------------
# 1. Load predictions
# ---------------------------------------------------------------------------

def load_predictions(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads predictions.parquet and returns validation and test DataFrames.

    The test set is kept separate and used only once for final reporting,
    consistent with the strict out-of-sample evaluation design.
    """
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    val  = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()

    print(f"  Validation : {len(val):,} rows")
    print(f"  Test       : {len(test):,} rows")
    return val, test


# ---------------------------------------------------------------------------
# 2. Compute realized imbalance cost per PTU
# ---------------------------------------------------------------------------

def compute_imbalance_cost(df: pd.DataFrame,
                           pred_col: str) -> pd.Series:
    """
    Computes realized imbalance cost per PTU for a given forecast column.

    Parameters
    ----------
    df       : DataFrame containing load_mwh, price_shortage, price_surplus
    pred_col : Name of the forecast column ('pred_baseline' or 'pred_cost')

    Returns
    -------
    pd.Series of per-PTU imbalance costs in EUR

    Settlement logic:
      Over-prediction  (forecast > actual): pay price_surplus  per MWh error
      Under-prediction (forecast < actual): pay price_shortage per MWh error

    Note: Prices are in EUR/MWh. Errors are in MWh. Cost is in EUR.
    Note: Prices and costs can be negative (revenue in favorable conditions).
    """
    error = df[pred_col] - df["load_mwh"]

    cost = np.where(
        error > 0,
        error * df["price_surplus"],
        np.abs(error) * df["price_shortage"]
    )
    return pd.Series(cost, index=df.index, name=f"cost_{pred_col}")


# ---------------------------------------------------------------------------
# 3. RQ1 — Economic performance table
# ---------------------------------------------------------------------------

def table_economic_performance(val: pd.DataFrame,
                                test: pd.DataFrame) -> pd.DataFrame:
    """
    RQ1: Compares realized imbalance costs between baseline and cost-sensitive
    models on both validation and test sets.

    Metrics:
      - Mean cost per PTU [EUR]
      - Total annual cost [EUR]
      - Cost reduction vs baseline [EUR and %]
    """
    rows = []
    for label, df in [("Validation (2024)", val), ("Test (2025)", test)]:
        for col in ["pred_baseline", "pred_cost"]:
            cost = compute_imbalance_cost(df, col)
            rows.append({
                "Split":          label,
                "Model":          "Baseline (MSE)" if "baseline" in col else "Cost-Sensitive",
                "Mean cost (EUR/PTU)": round(cost.mean(), 2),
                "Total cost (EUR)":    round(cost.sum(), 0),
                "N":              len(df),
            })

    result = pd.DataFrame(rows)

    # Add cost reduction columns
    for split in result["Split"].unique():
        mask = result["Split"] == split
        base_cost = result.loc[mask & (result["Model"] == "Baseline (MSE)"),
                                "Total cost (EUR)"].values[0]
        cost_s    = result.loc[mask & (result["Model"] == "Cost-Sensitive"),
                                "Total cost (EUR)"].values[0]
        reduction = base_cost - cost_s
        pct       = (reduction / base_cost * 100) if base_cost != 0 else np.nan
        result.loc[mask & (result["Model"] == "Cost-Sensitive"),
                   "Cost reduction (EUR)"] = round(reduction, 0)
        result.loc[mask & (result["Model"] == "Cost-Sensitive"),
                   "Cost reduction (%)"]   = round(pct, 2)

    return result


# ---------------------------------------------------------------------------
# 4. RQ2 — Forecast bias and error pattern table
# ---------------------------------------------------------------------------

def table_forecast_bias(val: pd.DataFrame,
                         test: pd.DataFrame) -> pd.DataFrame:
    """
    RQ2: Quantifies systematic forecast bias and directional error patterns
    for both models on validation and test sets.

    Metrics:
      - Mean forecast error (bias) [MWh] — positive = over-prediction
      - Over-prediction rate [%]
      - Under-prediction rate [%]
      - Mean absolute error [MWh]
      - RMSE [MWh]
    """
    rows = []
    for label, df in [("Validation (2024)", val), ("Test (2025)", test)]:
        for col in ["pred_baseline", "pred_cost"]:
            error = df[col] - df["load_mwh"]
            rows.append({
                "Split":              label,
                "Model":              "Baseline (MSE)" if "baseline" in col else "Cost-Sensitive",
                "Mean bias (MWh)":    round(error.mean(), 2),
                "Over-pred rate (%)": round((error > 0).mean() * 100, 1),
                "Under-pred rate (%)":round((error < 0).mean() * 100, 1),
                "MAE (MWh)":          round(error.abs().mean(), 2),
                "RMSE (MWh)":         round(np.sqrt((error**2).mean()), 2),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. RQ2 — Regime analysis table
# ---------------------------------------------------------------------------

def table_regime_analysis(test: pd.DataFrame) -> pd.DataFrame:
    """
    RQ2: Examines how forecast bias and imbalance costs differ across
    market regimes on the test set (2025).

    Regimes:
      Peak vs off-peak     : hour 08:00–20:00 vs remainder
      High vs normal price : |price_shortage| above/below 75th percentile
      Dual vs single pricing : dual_pricing flag from dataset (regulation_state == 2)

    This reveals whether the cost-sensitive model's behaviour differs
    most in the intervals where it matters most economically.
    """
    # Compute high-price threshold on the TRAINING set's price distribution,
    # then apply this fixed value to the test set. This avoids defining the
    # regime endogenously to the data being analysed.
    train = pd.read_parquet(DATA_PATH).query("split == 'train'")
    threshold = train["price_shortage"].abs().quantile(HIGH_PRICE_QUANTILE)

    test = test.copy()
    test["peak"]       = test["hour"].isin(PEAK_HOURS)
    test["high_price"] = test["price_shortage"].abs() > threshold

    # Merge dual_pricing flag from the full dataset
    if "dual_pricing" not in test.columns:
        ds = pd.read_parquet(DATA_PATH)[["timestamp", "dual_pricing"]]
        ds["timestamp"] = pd.to_datetime(ds["timestamp"])
        test = test.merge(ds, on="timestamp", how="left")

    rows = []
    for regime_col, regime_labels in [
        ("peak",          {True: "Peak (08-20h)", False: "Off-peak"}),
        ("high_price",    {True: f"High price (>{threshold:.0f} EUR/MWh)",
                           False: "Normal price"}),
        ("dual_pricing",  {1: "Dual pricing", 0: "Single pricing"}),
    ]:
        for regime_val, regime_label in regime_labels.items():
            subset = test[test[regime_col] == regime_val]
            for col in ["pred_baseline", "pred_cost"]:
                error = subset[col] - subset["load_mwh"]
                cost  = compute_imbalance_cost(subset, col)
                rows.append({
                    "Regime":             regime_label,
                    "Model":              "Baseline (MSE)" if "baseline" in col else "Cost-Sensitive",
                    "N":                  len(subset),
                    "Mean bias (MWh)":    round(error.mean(), 2),
                    "Over-pred rate (%)": round((error > 0).mean() * 100, 1),
                    "Mean cost (EUR/PTU)":round(cost.mean(), 2),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Figure 1 — Residual distributions (RQ2)
# ---------------------------------------------------------------------------

def fig_residuals(test: pd.DataFrame):
    """
    Figure 1: Residual (error) distribution histograms for both models
    on the test set.

    Required by DSS rubric: error pattern visualization.
    Shows whether the cost-sensitive model shifts the error distribution
    relative to the MSE baseline.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    configs = [
        ("pred_baseline", "Baseline (MSE)", "#2c7bb6"),
        ("pred_cost",     "Cost-Sensitive", "#d7191c"),
    ]

    for ax, (col, label, color) in zip(axes, configs):
        error = test[col] - test["load_mwh"]
        ax.hist(error, bins=80, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(0,           color="black",  lw=1.5, linestyle="--",
                   label="Zero error")
        ax.axvline(error.mean(), color=color,   lw=1.5, linestyle="-",
                   label=f"Mean bias = {error.mean():.1f} MWh")
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Forecast error (MWh)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=10)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}"))

    fig.suptitle("Residual Distributions — Test Set (2025)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_RESIDUALS, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_RESIDUALS}")


# ---------------------------------------------------------------------------
# 7. Figure 2 — Predicted vs actual scatter (DSS rubric requirement)
# ---------------------------------------------------------------------------

def fig_pred_vs_actual(test: pd.DataFrame):
    """
    Figure 2: Predicted vs actual load scatter plots for both models.

    Explicitly required by the DSS rubric as an error pattern visualization
    for regression tasks. A perfect forecast lies on the 45-degree diagonal.
    Systematic deviations reveal bias patterns.

    Plots a random sample of 2,000 points for visual clarity.
    """
    sample = test.sample(n=min(2000, len(test)), random_state=42)
    lims   = (test["load_mwh"].min() - 100,
              test["load_mwh"].max() + 100)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    configs = [
        ("pred_baseline", "Baseline (MSE)", "#2c7bb6"),
        ("pred_cost",     "Cost-Sensitive", "#d7191c"),
    ]

    for ax, (col, label, color) in zip(axes, configs):
        ax.scatter(sample["load_mwh"], sample[col],
                   alpha=0.3, s=8, color=color)
        ax.plot(lims, lims, color="black", lw=1.5,
                linestyle="--", label="Perfect forecast")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Actual load (MWh)", fontsize=11)
        ax.set_ylabel("Predicted load (MWh)", fontsize=11)
        ax.legend(fontsize=10)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}"))

    fig.suptitle("Predicted vs Actual Load — Test Set (2025)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_PRED_ACTUAL, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_PRED_ACTUAL}")


# ---------------------------------------------------------------------------
# 8. Figure 3 — Regime bias breakdown (RQ2)
# ---------------------------------------------------------------------------

def fig_regime_bias(test: pd.DataFrame):
    """
    Figure 3: Mean forecast bias by market regime and model.

    Grouped bar chart comparing mean bias (MWh) across peak/off-peak
    and high/normal imbalance price regimes for both models.
    Reveals whether the cost-sensitive model's bias shift is concentrated
    in the economically most relevant intervals.
    """
    train = pd.read_parquet(DATA_PATH).query("split == 'train'")
    threshold = train["price_shortage"].abs().quantile(HIGH_PRICE_QUANTILE)
    test = test.copy()
    test["peak"]       = test["hour"].isin(PEAK_HOURS)
    test["high_price"] = test["price_shortage"].abs() > threshold

    regimes = {
        "Peak":       test[test["peak"]],
        "Off-peak":   test[~test["peak"]],
        "High price": test[test["high_price"]],
        "Normal price": test[~test["high_price"]],
    }

    x      = np.arange(len(regimes))
    width  = 0.35
    labels = list(regimes.keys())

    bias_base = [regimes[r]["pred_baseline"].sub(
                  regimes[r]["load_mwh"]).mean() for r in labels]
    bias_cost = [regimes[r]["pred_cost"].sub(
                  regimes[r]["load_mwh"]).mean() for r in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, bias_base, width, label="Baseline (MSE)",
           color="#2c7bb6", alpha=0.8)
    ax.bar(x + width/2, bias_cost, width, label="Cost-Sensitive",
           color="#d7191c", alpha=0.8)
    ax.axhline(0, color="black", lw=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean forecast bias (MWh)", fontsize=11)
    ax.set_title("Mean Forecast Bias by Market Regime — Test Set (2025)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_REGIME_BIAS, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_REGIME_BIAS}")


# ---------------------------------------------------------------------------
# 9. Figure 4 — Cumulative imbalance cost over test year (RQ1)
# ---------------------------------------------------------------------------

def fig_cumulative_cost(test: pd.DataFrame):
    """
    Figure 4: Cumulative realized imbalance cost over the 2025 test year.

    Shows the trajectory of cost accumulation for both models over the full
    test year. A cost-sensitive model that outperforms the baseline will show
    a lower cumulative cost line throughout the year.
    """
    test = test.sort_values("timestamp").copy()

    cost_base = compute_imbalance_cost(test, "pred_baseline").cumsum()
    cost_sens = compute_imbalance_cost(test, "pred_cost").cumsum()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test["timestamp"], cost_base, color="#2c7bb6", lw=1.5,
            label="Baseline (MSE)")
    ax.plot(test["timestamp"], cost_sens, color="#d7191c", lw=1.5,
            label="Cost-Sensitive")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative imbalance cost (EUR)", fontsize=11)
    ax.set_title("Cumulative Realized Imbalance Cost — Test Set (2025)",
                 fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"€{x:,.0f}"))
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_CUMCOST, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_CUMCOST}")


# ---------------------------------------------------------------------------
# 10. Figure 6 — Cumulative cost split by dual/single pricing (RQ1)
# ---------------------------------------------------------------------------

def fig_cumulative_cost_by_regime(test: pd.DataFrame):
    """
    Figure 6: Cumulative realized imbalance cost split by pricing regime.

    Two panels:
      Left  — Dual-pricing intervals only (where prices diverge)
      Right — Single-pricing intervals only (where prices are equal)

    Visually demonstrates that the cost-sensitive model's advantage is
    concentrated in dual-pricing intervals, while single-pricing intervals
    show nearly identical trajectories.
    """
    # Merge dual_pricing flag
    if "dual_pricing" not in test.columns:
        ds = pd.read_parquet(DATA_PATH)[["timestamp", "dual_pricing"]]
        ds["timestamp"] = pd.to_datetime(ds["timestamp"])
        test = test.merge(ds, on="timestamp", how="left")

    test = test.sort_values("timestamp").copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (regime_val, regime_label) in zip(axes, [
        (1, "Dual Pricing Intervals"),
        (0, "Single Pricing Intervals"),
    ]):
        subset = test[test["dual_pricing"] == regime_val].copy()
        cost_base = compute_imbalance_cost(subset, "pred_baseline").cumsum()
        cost_sens = compute_imbalance_cost(subset, "pred_cost").cumsum()

        ax.plot(subset["timestamp"], cost_base, color="#2c7bb6", lw=1.5,
                label="Baseline (MSE)")
        ax.plot(subset["timestamp"], cost_sens, color="#d7191c", lw=1.5,
                label="Cost-Sensitive")

        reduction = cost_base.iloc[-1] - cost_sens.iloc[-1]
        ax.set_title(f"{regime_label} (n={len(subset):,})\n"
                     f"Reduction: €{reduction:,.0f}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Cumulative imbalance cost (EUR)", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"€{x:,.0f}"))
        ax.legend(fontsize=10)

    fig.suptitle("Cumulative Imbalance Cost by Pricing Regime — Test Set (2025)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_CUMCOST_REGIME, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_CUMCOST_REGIME}")


# ---------------------------------------------------------------------------
# 11. Figure 6 — Empirical shortage vs surplus price distributions
# ---------------------------------------------------------------------------

def fig_price_distributions():
    """
    Figure 6: empirical distributions of |p_shortage| vs |p_surplus|,
    decomposed by partition (train / validation / test). Restricted to
    dual-pricing PTUs because in single-pricing intervals the two
    settlement prices coincide by construction.

    Anchors the mechanism argument in Section 6.2 of the thesis: the
    upward bias shift of the cost-sensitive model is the rational
    response to a stable, empirically visible asymmetry between
    shortage and surplus prices (mean ratio 1.76--1.88x across all
    three non-overlapping partitions).
    """
    ds = pd.read_parquet(DATA_PATH)
    ds = ds[ds["dual_pricing"] == 1]

    partitions = [
        ("Training (2022–2023)", ds["split"] == "train"),
        ("Validation (2024)",         ds["split"] == "validation"),
        ("Test (2025)",               ds["split"] == "test"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    bins = np.logspace(0, 4, 60)  # 1 to 10,000 EUR/MWh on log axis

    for ax, (label_text, mask) in zip(axes, partitions):
        sub = ds[mask]
        short_all = sub["price_shortage"].abs().values
        surp_all  = sub["price_surplus"].abs().values

        # Ratio is computed on the full population (matching the figures
        # reported in Section 4.6 of the methodology). Log-scale plotting
        # only filters zeros for the histogram below.
        m_short = short_all.mean()
        m_surp  = surp_all.mean()
        ratio   = m_short / m_surp

        short = short_all[short_all > 0]
        surp  = surp_all[surp_all > 0]

        ax.hist(short, bins=bins, density=True,
                color="#d7191c", alpha=0.55, edgecolor="#7f0c11",
                linewidth=0.4,
                label=r"$|p^{\mathrm{shortage}}|$")
        ax.hist(surp, bins=bins, density=True,
                color="#2c7bb6", alpha=0.55, edgecolor="#1a4b6e",
                linewidth=0.4,
                label=r"$|p^{\mathrm{surplus}}|$")

        # Mean lines (visual anchor for the ratio)
        ax.axvline(m_short, color="#d7191c", linestyle="--",
                   linewidth=1.3, alpha=0.95)
        ax.axvline(m_surp,  color="#2c7bb6", linestyle="--",
                   linewidth=1.3, alpha=0.95)

        ax.set_xscale("log")
        ax.set_xlim(1, 5000)
        ax.set_title(f"{label_text}\nmean ratio = {ratio:.2f}×",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Absolute settlement price (EUR/MWh)",
                      fontsize=11)
        ax.grid(True, which="both", alpha=0.2)
        if ax is axes[0]:
            ax.set_ylabel("Density", fontsize=11)
            ax.legend(loc="upper right", fontsize=10)

    fig.suptitle(
        "Empirical distribution of Dutch settlement prices "
        "in dual-pricing PTUs",
        fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_PRICE_DIST, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_PRICE_DIST}")


# ---------------------------------------------------------------------------
# 12. Main
# ---------------------------------------------------------------------------

def main():
    print("Loading predictions...")
    val, test = load_predictions(PREDS_PATH)

    print("\n[RQ1] Economic performance...")
    t1 = table_economic_performance(val, test)
    print(t1.to_string(index=False))

    print("\n[RQ2] Forecast bias...")
    t2 = table_forecast_bias(val, test)
    print(t2.to_string(index=False))

    print("\n[RQ2] Regime analysis (test set only)...")
    t3 = table_regime_analysis(test)
    print(t3.to_string(index=False))

    print("\n[RQ3] Trade-off summary already in Table 1 + Table 2 above.")

    # Save all tables to CSV
    combined = pd.concat([
        t1.assign(Table="RQ1 Economic Performance"),
        t2.assign(Table="RQ2 Forecast Bias"),
        t3.assign(Table="RQ2 Regime Analysis"),
    ], ignore_index=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"\n  Saved: {OUT_CSV}")

    print("\nGenerating figures...")
    fig_residuals(test)
    fig_pred_vs_actual(test)
    fig_regime_bias(test)
    fig_cumulative_cost(test)
    fig_cumulative_cost_by_regime(test)
    fig_price_distributions()

    print("\nDone. All results saved to thesis folder.")


if __name__ == "__main__":
    main()