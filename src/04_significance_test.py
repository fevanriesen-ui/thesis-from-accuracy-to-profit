"""
04_significance_test.py
═══════════════════════════════════════════════════════════════════════════════
Statistical Significance Testing — MSc Thesis: From Accuracy to Profit
F.E. van Riesen | Tilburg University | Data Science & Society | 2026
═══════════════════════════════════════════════════════════════════════════════

PURPOSE
-------
Tests whether the per-PTU imbalance cost difference between the baseline and
cost-sensitive model on the 2025 test set is statistically significant.

Two tests are reported:
  1. Paired t-test       — parametric, assumes normality
  2. Wilcoxon signed-rank — non-parametric, robust to outliers

The Wilcoxon test is the primary test for this data because imbalance costs
have heavy-tailed distributions driven by extreme price events, violating the
normality assumption required by the t-test. Both are reported for completeness.

Also produces (all reported in significance_results.txt for reproducibility):
  - Descriptive statistics on per-PTU cost differences
  - Dual / single pricing breakdown on BOTH validation (2024) and test (2025),
    with high-precision p-values and reduction expressed as % of |baseline|
  - Dual-pricing share per partition (8.2% / 16.8% / 27.3% / 15.1%)
  - Shortage/surplus price ratio per partition (asymmetry that drives the
    expectile shift documented in the Discussion)
  - Regime-conditional bias shift on the test set (peak / high price / dual)
  - Load distribution table by year (explains systematic over-prediction bias)

INPUTS
------
  predictions.parquet   — output of 02_train_models.py (Option A / default params)
  dataset.parquet       — output of 01_build_dataset.py

OUTPUTS
-------
  significance_results.txt   — full test results, saved to thesis folder

USAGE
-----
  python 04_significance_test.py
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDS_PATH   = os.path.join(PROJECT_ROOT, "outputs", "predictions.parquet")
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "processed", "dataset.parquet")
OUT_TXT      = os.path.join(PROJECT_ROOT, "outputs", "significance_results.txt")

# ---------------------------------------------------------------------------
# 1. Imbalance cost function (same as 03_evaluate.py)
# ---------------------------------------------------------------------------

def compute_imbalance_cost(df: pd.DataFrame, pred_col: str) -> pd.Series:
    """
    Computes realized imbalance cost per PTU.
    Over-prediction  (forecast > actual): cost = error × price_surplus
    Under-prediction (forecast ≤ actual): cost = |error| × price_shortage
    """
    error = df[pred_col] - df["load_mwh"]
    cost  = np.where(
        error > 0,
        error * df["price_surplus"],
        np.abs(error) * df["price_shortage"]
    )
    return pd.Series(cost, index=df.index)


def fmt_p(p: float) -> str:
    """Format p-values: scientific notation for tiny values, 4 dp otherwise."""
    if p < 1e-4:
        return f"{p:.3e}"
    return f"{p:.4f}"


def regime_breakdown(label_top: str, df: pd.DataFrame, log) -> dict:
    """
    Prints baseline-vs-cost-sensitive comparison split by dual / single pricing
    for a single partition (validation or test). Reports reduction expressed
    as a percentage of the *absolute* baseline cost so the sign is interpretable
    even when the baseline cost is negative (BRP revenue).
    """
    out = {}
    for regime_label, subset in [
        ("Dual pricing",   df[df["dual_pricing"] == 1]),
        ("Single pricing", df[df["dual_pricing"] == 0]),
    ]:
        cb = compute_imbalance_cost(subset, "pred_baseline")
        cs = compute_imbalance_cost(subset, "pred_cost")
        diff = cb - cs
        reduction = diff.sum()
        base_abs = abs(cb.sum())
        pct_abs = (reduction / base_abs * 100) if base_abs > 0 else np.nan
        log(f"\n  {label_top} — {regime_label} (n={len(subset):,}):")
        log(f"    Baseline total cost  : € {cb.sum():>15,.0f}")
        log(f"    Cost-sens total cost : € {cs.sum():>15,.0f}")
        log(f"    Reduction            : € {reduction:>15,.0f}  "
            f"({pct_abs:+.2f}% of |baseline|)")
        log(f"    Mean reduction / PTU : €{diff.mean():,.2f}")
        if len(subset) > 10:
            w_s, p_w = stats.wilcoxon(cb, cs)
            t_s, p_t = stats.ttest_rel(cb, cs)
            d_s = diff.mean() / diff.std() if diff.std() > 0 else 0.0
            log(f"    Wilcoxon: W={w_s:.0f}, p={fmt_p(p_w)}"
                f"{'  ← significant' if p_w < 0.05 else ''}")
            log(f"    t-test  : t={t_s:.4f}, p={fmt_p(p_t)}"
                f"{'  ← significant' if p_t < 0.05 else ''}")
            log(f"    Cohen's d: {d_s:.4f}")
            out[regime_label] = dict(
                n=len(subset), reduction=reduction, pct_abs=pct_abs,
                p_wilcoxon=p_w, p_ttest=p_t, cohen_d=d_s,
            )
    return out


# ---------------------------------------------------------------------------
# 2. Main
# ---------------------------------------------------------------------------

def main():
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 70)
    log("STATISTICAL SIGNIFICANCE TESTS")
    log("Thesis: From Accuracy to Profit — F.E. van Riesen (2026)")
    log("Option A: Default hyperparameters (max_depth=6, lr=0.05)")
    log("=" * 70)

    # --- Load predictions and merge dual_pricing flag ---
    preds = pd.read_parquet(PREDS_PATH)
    preds["timestamp"] = pd.to_datetime(preds["timestamp"])

    ds = pd.read_parquet(DATA_PATH)
    ds["timestamp"] = pd.to_datetime(ds["timestamp"])
    preds = preds.merge(
        ds[["timestamp", "dual_pricing"]], on="timestamp", how="left",
    )

    val  = preds[preds["split"] == "validation"].copy()
    test = preds[preds["split"] == "test"].copy()
    log(f"\nValidation set: {len(val):,} PTU intervals (2024)")
    log(f"Test set      : {len(test):,} PTU intervals (2025)")

    # --- Compute per-PTU costs ---
    cost_base = compute_imbalance_cost(test, "pred_baseline")
    cost_sens = compute_imbalance_cost(test, "pred_cost")
    cost_diff = cost_base - cost_sens  # positive = baseline costs more

    log("\n--- Per-PTU Cost Difference (Baseline minus Cost-Sensitive) ---")
    log(f"  Mean difference      : €{cost_diff.mean():,.2f} per PTU")
    log(f"  Std deviation        : €{cost_diff.std():,.2f}")
    log(f"  Median difference    : €{cost_diff.median():,.2f}")
    log(f"  Intervals where baseline costs more : "
        f"{(cost_diff > 0).sum():,} ({(cost_diff > 0).mean()*100:.1f}%)")
    log(f"  Intervals where cost-sensitive costs more : "
        f"{(cost_diff < 0).sum():,} ({(cost_diff < 0).mean()*100:.1f}%)")
    log(f"  Total cost reduction (baseline - cost-sensitive): "
        f"€{cost_diff.sum():,.0f}")

    # --- Paired t-test ---
    log("\n--- Test 1: Paired t-test (parametric) ---")
    t_stat, p_val_t = stats.ttest_rel(cost_base, cost_sens)
    log(f"  H0: Mean per-PTU cost is equal across models")
    log(f"  t-statistic : {t_stat:.4f}")
    log(f"  p-value     : {fmt_p(p_val_t)}")
    if p_val_t < 0.05:
        log(f"  Result      : SIGNIFICANT at α=0.05 — reject H0")
    else:
        log(f"  Result      : NOT significant at α=0.05 — fail to reject H0")
    log(f"  Note: t-test assumes normality. Imbalance costs have heavy-tailed")
    log(f"        distributions; the Wilcoxon test below is more appropriate.")

    # --- Wilcoxon signed-rank test ---
    log("\n--- Test 2: Wilcoxon Signed-Rank Test (non-parametric) ---")
    w_stat, p_val_w = stats.wilcoxon(cost_base, cost_sens)
    log(f"  H0: Median per-PTU cost difference is zero")
    log(f"  W-statistic : {w_stat:.0f}")
    log(f"  p-value     : {fmt_p(p_val_w)}")
    if p_val_w < 0.05:
        log(f"  Result      : SIGNIFICANT at α=0.05 — reject H0")
    else:
        log(f"  Result      : NOT significant at α=0.05 — fail to reject H0")
    log(f"  Note: Non-parametric test, robust to outliers and non-normality.")
    log(f"        Preferred test for heavy-tailed imbalance cost distributions.")

    # --- Normality check (justifies Wilcoxon preference) ---
    log("\n--- Normality Check on Cost Differences ---")
    _, p_shapiro = stats.shapiro(cost_diff.sample(min(5000, len(cost_diff)),
                                                   random_state=42))
    log(f"  Shapiro-Wilk test on sample of {min(5000, len(cost_diff)):,}: "
        f"p={p_shapiro:.4f}")
    if p_shapiro < 0.05:
        log(f"  Distribution is NOT normal (p<0.05) → Wilcoxon is appropriate")
    else:
        log(f"  Distribution appears normal → both tests are appropriate")

    # --- Effect size ---
    log("\n--- Effect Size ---")
    cohens_d = cost_diff.mean() / cost_diff.std()
    log(f"  Cohen's d : {cohens_d:.4f}")
    if abs(cohens_d) < 0.2:
        label = "negligible"
    elif abs(cohens_d) < 0.5:
        label = "small"
    elif abs(cohens_d) < 0.8:
        label = "medium"
    else:
        label = "large"
    log(f"  Interpretation : {label} effect size")

    # --- Dual vs single pricing breakdown (validation AND test) ---
    log("\n--- Dual vs Single Pricing Breakdown ---")
    log("(Asymmetric loss only differs from symmetric in dual-pricing intervals.")
    log(" Reductions are expressed as % of |baseline| so the sign is")
    log(" interpretable even when baseline cost is negative = BRP revenue.)")

    test_stats = regime_breakdown("Test (2025)",       test, log)
    val_stats  = regime_breakdown("Validation (2024)", val,  log)

    # --- Dataset structure summary (reproduces §5.1 / Discussion claims) ---
    log("\n--- Dataset Structure: dual-pricing share per partition ---")
    log("(Reproduces the regime-share figures cited in Methodology and Discussion)")
    for name, mask in [
        ("Train  2022-2023", ds["split"] == "train"),
        ("Val    2024",      ds["split"] == "validation"),
        ("Test   2025",      ds["split"] == "test"),
        ("Full   dataset",   pd.Series([True] * len(ds))),
    ]:
        sub = ds[mask]
        share = sub["dual_pricing"].mean() * 100
        log(f"  {name:18s}  dual = {share:5.2f}%  "
            f"({sub['dual_pricing'].sum():,}/{len(sub):,})")

    # --- Shortage / Surplus price ratio (justifies the upward bias shift) ---
    log("\n--- Shortage/Surplus Price Ratio (dual-pricing intervals only) ---")
    log("(Confirms the asymmetry that drives the cost-sensitive model's upward")
    log(" bias shift; mean |price_shortage| ÷ mean |price_surplus|)")
    for name, mask in [
        ("Train  2022-2023", (ds["split"] == "train")      & (ds["dual_pricing"] == 1)),
        ("Val    2024",      (ds["split"] == "validation") & (ds["dual_pricing"] == 1)),
        ("Test   2025",      (ds["split"] == "test")       & (ds["dual_pricing"] == 1)),
    ]:
        sub = ds[mask]
        m_short = sub["price_shortage"].abs().mean()
        m_surp  = sub["price_surplus"].abs().mean()
        ratio   = m_short / m_surp if m_surp > 0 else np.nan
        log(f"  {name:18s}  short=€{m_short:7.2f}/MWh  "
            f"surp=€{m_surp:7.2f}/MWh  ratio={ratio:.3f}×")

    # --- Regime-conditional bias shift on test set ---
    log("\n--- Regime-Conditional Bias Shift on Test 2025 ---")
    log("(Cost-sensitive minus baseline mean forecast error, by regime.")
    log(" Confirms that the expectile shift concentrates where dual-pricing")
    log(" and large absolute prices are densest.)")
    test_b = test.copy()
    test_b["hour"] = test_b["timestamp"].dt.hour
    test_b["peak"] = test_b["hour"].between(8, 20)
    train_threshold = ds.loc[ds["split"] == "train",
                              "price_shortage"].abs().quantile(0.75)
    test_b["high_price"] = test_b["price_shortage"].abs() > train_threshold
    log(f"  (high-price threshold = €{train_threshold:.0f}/MWh, "
        f"75th pct of |price_shortage| on train)")
    for name, mask in [
        ("Peak (08-20h)",   test_b["peak"]),
        ("Off-peak",        ~test_b["peak"]),
        ("High price",      test_b["high_price"]),
        ("Normal price",    ~test_b["high_price"]),
        ("Dual pricing",    test_b["dual_pricing"] == 1),
        ("Single pricing",  test_b["dual_pricing"] == 0),
    ]:
        sub = test_b[mask]
        bias_b = (sub["pred_baseline"] - sub["load_mwh"]).mean()
        bias_c = (sub["pred_cost"]     - sub["load_mwh"]).mean()
        log(f"  {name:16s}  n={len(sub):>6,d}  "
            f"baseline={bias_b:+8.2f}  cost-sens={bias_c:+8.2f}  "
            f"shift={bias_c - bias_b:+7.2f} MWh")

    # --- Load distribution by year ---
    log("\n--- Load Distribution by Year ---")
    log("(Explains systematic over-prediction bias: test load is lower than training)")
    load_table = ds.groupby(ds["timestamp"].dt.year).agg(
        mean_load_mwh=("load_mwh", "mean"),
        std_load_mwh=("load_mwh", "std"),
        n_obs=("load_mwh", "count"),
    ).round(1)
    log(f"\n{load_table.to_string()}")

    train_mean = ds.loc[ds["split"] == "train",      "load_mwh"].mean()
    val_mean   = ds.loc[ds["split"] == "validation", "load_mwh"].mean()
    test_mean  = ds.loc[ds["split"] == "test",       "load_mwh"].mean()
    log(f"\n  Train mean  (2022-2023) : {train_mean:.1f} MWh")
    log(f"  Val mean   (2024)       : {val_mean:.1f} MWh")
    log(f"  Test mean  (2025)       : {test_mean:.1f} MWh")
    log(f"\n  Load shift train→test   : {test_mean - train_mean:.1f} MWh "
        f"({(test_mean / train_mean - 1) * 100:.1f}%)")

    # --- Summary for thesis ---
    log("\n" + "=" * 70)
    log("SUMMARY FOR THESIS REPORTING")
    log("=" * 70)
    log(f"  Primary test (Wilcoxon): p={fmt_p(p_val_w)} "
        f"{'→ significant' if p_val_w < 0.05 else '→ not significant'}")
    log(f"  Secondary test (t-test): p={fmt_p(p_val_t)} "
        f"{'→ significant' if p_val_t < 0.05 else '→ not significant'}")
    log(f"  Effect size (Cohen's d): {cohens_d:.4f} ({label})")
    log(f"  Total cost reduction: €{cost_diff.sum():,.0f}")
    log(f"  Mean cost reduction per PTU: €{cost_diff.mean():,.2f}")
    log(f"  Test set: {len(test):,} intervals (full calendar year 2025)")

    dp = test_stats.get("Dual pricing", {})
    if dp:
        log(f"  Dual-pricing subset (n={dp['n']:,}): "
            f"Wilcoxon p={fmt_p(dp['p_wilcoxon'])} "
            f"{'→ significant' if dp['p_wilcoxon'] < 0.05 else '→ not significant'}, "
            f"reduction €{dp['reduction']:,.0f} ({dp['pct_abs']:+.2f}% of |baseline|)")
    dp_val = val_stats.get("Dual pricing", {})
    if dp_val:
        log(f"  Validation dual subset (n={dp_val['n']:,}): "
            f"Wilcoxon p={fmt_p(dp_val['p_wilcoxon'])}, "
            f"reduction €{dp_val['reduction']:,.0f} "
            f"({dp_val['pct_abs']:+.2f}% of |baseline|)")

    # --- Save to file ---
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines))
    log(f"\n  Saved to: {OUT_TXT}")


if __name__ == "__main__":
    main()
