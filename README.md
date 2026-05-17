# From Accuracy to Profit
### Cost-Sensitive Load Forecasting and Forecast Bias under the Dutch Imbalance Settlement Regime

**MSc Thesis, Data Science & Society — Tilburg University, 2026**
F.E. van Riesen

This repository contains the full empirical pipeline of the thesis: data
preparation, model training, evaluation, and significance testing for a
controlled comparison between an MSE-trained and a cost-sensitive XGBoost
load forecaster on Dutch transmission-system data (2022–2025).

---

## Research question

> To what extent does training short-term load forecasting models on
> asymmetric Dutch imbalance settlement prices reduce realised imbalance
> costs for Balance Responsible Parties (BRPs) relative to an MSE
> benchmark, and what systematic forecast biases emerge as a consequence?

Two XGBoost models are trained on identical features, hyperparameters, and
training data — only the loss function differs. The MSE baseline minimises
symmetric quadratic error; the cost-sensitive model weights each squared
error by the realised per-PTU Dutch settlement price applicable to the
direction of the error.

Headline result on the 2025 test year: in the 9,578 dual-pricing PTUs
(where the two losses imply distinct optimal forecasts), the cost-sensitive
model reduces realised imbalance settlement cost by 7.06% of the absolute
baseline (Wilcoxon p = 4.51 × 10⁻⁶), at the cost of a 2.7% MAE / 2.2% RMSE
degradation across all 35,036 test intervals.

---

## Pipeline

The pipeline consists of four scripts executed in order:

| Step | Script | Output |
|------|--------|--------|
| 1 | `src/01_build_dataset.py` | `data/processed/dataset.parquet` (139,568 rows × 19 cols, 15-min PTU) |
| 2 | `src/02_train_models.py` | `models/model_baseline.json`, `models/model_cost.json`, `outputs/predictions.parquet` |
| 3 | `src/03_evaluation.py` | `outputs/results_summary.csv`, `outputs/figures/fig{1–6}.png` |
| 4 | `src/04_significance_test.py` | `outputs/significance_results.txt` |

Run the full pipeline with:

```bash
python main.py                 # all four steps
python main.py --from 2        # skip dataset rebuild, run from step 2
python main.py 3 4             # only evaluation + significance tests
```

---

## Setup

Requires Python 3.13 (developed on macOS / darwin; should run on Linux).
XGBoost requires the OpenMP runtime on macOS — install with
`brew install libomp` if needed.

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data

Raw data is **not** included in this repository. Place the source CSVs
under `data/raw/` before running step 1. Full download instructions and
the expected filename conventions are in [data/README.md](data/README.md).

Briefly, the pipeline expects:

- TenneT Transparency Portal (https://www.tennet.eu/dutch-transparency-portal/):
  `metered_injections_<YYYY-YYYY>.csv`,
  `settlement_prices_<YYYY-YYYY>.csv`,
  `settled_imbalance_volumes_<YYYY-YYYY>.csv` — one file per year, 2022–2025
- KNMI Climate Data Portal (https://daggegevens.knmi.nl/): hourly
  observations for station De Bilt (260), saved as a single
  `knmi_weather.csv`

Both sources are public and free; neither contains personal or sensitive
data. See the DSECT statement in the thesis for the full data-source
disclosure.

---

## Repository structure

```
.
├── README.md                          ← this file
├── LICENSE                            ← MIT (code only)
├── requirements.txt
├── .gitignore
├── main.py                            ← pipeline runner
├── src/
│   ├── 01_build_dataset.py
│   ├── 02_train_models.py
│   ├── 03_evaluation.py
│   └── 04_significance_test.py
├── data/
│   ├── README.md                      ← raw-data download instructions
│   ├── raw/                           ← TenneT + KNMI CSVs (not committed)
│   └── processed/
│       └── dataset.parquet            ← step-1 output (139,568 × 19)
├── models/
│   ├── model_baseline.json            ← step-2 output (MSE model)
│   └── model_cost.json                ← step-2 output (cost-sensitive model)
└── outputs/
    ├── predictions.parquet            ← step-2 output (val + test predictions)
    ├── results_summary.csv            ← step-3 output (Tables 1–4)
    ├── significance_results.txt       ← step-4 output (full test report)
    └── figures/
        ├── fig1_residuals.png
        ├── fig2_pred_vs_actual.png
        ├── fig3_regime_bias.png
        ├── fig4_cost_over_time.png
        └── fig6_cost_by_regime.png
```

The pipeline output artefacts under `models/` and `outputs/` are committed
so that the results reported in the thesis can be verified without
re-running the full training pipeline. Rebuilding from scratch takes
approximately 5–10 minutes on a modern laptop.

---

## Reproducibility

The pipeline is deterministic. Every random component is seeded with
`random_state=42` (XGBoost) and significance tests are exact (Wilcoxon
signed-rank, paired t-test). Re-running on the same inputs reproduces the
numbers reported in the thesis to the cent.

Strict temporal partitions are used throughout to avoid look-ahead and
dependence leakage:

- **Training:** 2022–2023 (n ≈ 69,400 PTUs)
- **Validation:** 2024 (n = 35,132 PTUs) — used for early stopping only
- **Test:** 2025 (n = 35,036 PTUs) — held out for one-time final evaluation

Settlement prices (`price_shortage`, `price_surplus`) are deliberately
excluded from the feature matrix and enter only through the cost-sensitive
model's loss-function closure on the training partition, preventing data
leakage from the evaluation regime into model training. See Section 4.2
of the thesis for the full leakage analysis.

---

## Citation

If you use this code or build on this work, please cite the thesis:

```bibtex
@mastersthesis{vanRiesen2026,
  author = {van Riesen, F. E.},
  title  = {From Accuracy to Profit: Cost-Sensitive Load Forecasting and
            Forecast Bias under the Dutch Imbalance Settlement Regime},
  school = {Tilburg University},
  year   = {2026},
  type   = {MSc Thesis, Data Science \& Society},
}
```

---

## License

Code in this repository is released under the [MIT License](LICENSE).
The thesis text and figures are © 2026 F.E. van Riesen; redistribution
requires permission. Raw datasets remain subject to the terms of use of
their respective upstream providers (TenneT Transparency Portal; KNMI
Climate Data Portal).

---

## Acknowledgements

This thesis was supervised by dr. Eva Vanmassenhove at the Department of
Cognitive Science & Artificial Intelligence, School of Humanities and
Digital Sciences, Tilburg University.
