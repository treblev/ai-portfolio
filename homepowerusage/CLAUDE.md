# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EDA project analyzing personal home electricity consumption patterns using ~2 years of daily and ~3 years of monthly data exported from a utility provider. The data includes temperature readings to enable usage/cost correlation analysis.

## Data

All data lives in `data/`:

- `dailyUsage3_18_2024_to_4_17_2026.csv` / `dailyCost3_18_2024_to_4_17_2026.csv` — 763 rows each, columns: `Meter read date`, `Usage date`, `Total kWh`/`Total cost ($)`, `High temperature (F)`, `Low temperature (F)`
- `monthlyUsageMay2023_to_Mar2026.csv` / `monthlyCostMay2023_to_Mar2026.csv` — 36 rows each, columns: `Bill start date`, `Bill end date`, `Total kWh`/`Total cost ($)`

Daily files have temperature data; monthly files do not.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist yet, the core dependencies are: `pandas`, `numpy`, `plotly`, `seaborn`, `jupyter`, `duckdb`. Use 'python3' command instead of just 'python'.

## Running the Notebook

```bash
jupyter notebook eda.ipynb
# or
jupyter lab
```
