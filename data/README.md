# Data directory

This directory holds raw input data (`raw/`) and the processed dataset
that the pipeline writes (`processed/`).

## Layout

```
data/
├── raw/                 ← raw CSVs from TenneT + KNMI (not committed)
│   ├── metered_injections_<YYYY-YYYY>.csv
│   ├── settlement_prices_<YYYY-YYYY>.csv
│   ├── settled_imbalance_volumes_<YYYY-YYYY>.csv
│   └── knmi_weather.csv
└── processed/
    └── dataset.parquet  ← step-1 output (139,568 rows × 19 cols)
```

The raw CSVs total roughly 25 MB and are not committed; they are public
and re-downloadable from the upstream providers. See the instructions
below.

## TenneT Transparency Portal

For each of the four years 2022, 2023, 2024, and 2025, download:

- **Metered feed** (*Gemeten invoeding* / Measured Infeed) — the forecast
  target. Saved as `metered_injections_<YYYY-YYYY>.csv`.
- **Settlement prices** (per-PTU imbalance prices including the regulation
  state). Saved as `settlement_prices_<YYYY-YYYY>.csv`.
- **Settled imbalance volumes** (realised system imbalance per PTU; kept
  alongside the prices for completeness but not currently used by the
  pipeline as a feature). Saved as `settled_imbalance_volumes_<YYYY-YYYY>.csv`.

Source: https://www.tennet.eu/dutch-transparency-portal/ → Operation
section.

The pipeline globs files matching `metered_injections_*.csv`,
`settlement_prices_*.csv`, and `settled_imbalance_volumes_*.csv` under
`data/raw/`, so the exact filename suffix produced by the portal's
download dialog does not need to match the example names above.

## KNMI Climate Data Portal

Hourly observations for station **De Bilt (260)** for 2022–2025. The
pipeline expects a single CSV at `data/raw/knmi_weather.csv` with the
KNMI hourly export format (station, date, hour, wind_speed, temperature).

Source: https://daggegevens.knmi.nl/ — request hourly data for station
260 across the four-year window.

## License and use

Both datasets are public and free to use for research. They are not
redistributed in this repository; refer to the upstream providers'
terms of use for redistribution beyond personal/research use.
