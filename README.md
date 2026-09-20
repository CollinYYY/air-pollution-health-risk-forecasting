# Uncertainty-Aware Forecasting of Air Pollution and Health Risk

> Code-only portfolio edition of an exploratory research workflow. It contains no raw data, unpublished results, private paths, model artifacts, or figures.

## What this repository demonstrates

- Validation and concentration-unit standardisation for daily environmental data.
- Calculation of non-carcinogenic risk (HI) and carcinogenic risk (CR) from metal concentrations.
- Calendar, lagged, and strictly historical rolling features.
- Correlation screening and VIF-based collinearity reduction.
- Expanding-window TimeSeriesSplit evaluation rather than random train/test splits.
- CatBoost forecasting plus feature-importance export.

The original exploratory study compared additional deep-learning and stacked models. This public edition intentionally keeps one compact, reproducible core workflow instead of uploading redundant experimental scripts or unpublished outputs. It demonstrates data-processing and validation design; it is not a claim that research findings can be reproduced without the authorised data.

## Repository layout

```text
src/
  forecasting_pipeline.py         # compact end-to-end workflow
data/                             # intentionally empty; private inputs are ignored
outputs/                          # ignored generated artifacts
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/forecasting_pipeline.py \
  --input data/private_input.xlsx \
  --target PM2.5 \
  --output outputs
```

Use `--target HI` or `--target CR` to calculate the respective risk index before forecasting. The risk-factor parameters are retained from an exploratory-study configuration and should be reviewed against the target population and formal risk-assessment guideline before any applied use.

## Data availability

Original daily environmental, meteorological, exposure and health-risk tables are not included. They contain unpublished research material and may have usage restrictions. Do not commit private input files, generated metrics, model artifacts, figures, run logs, or manuscript drafts.
