# Uncertainty-Aware Forecasting of Air Pollution and Health Risk

Code-only portfolio edition of an exploratory environmental-health research workflow. No raw data, unpublished results, private paths, or model artifacts are included.

## Demonstrated workflow

- Daily environmental-data validation and concentration-unit standardisation
- HI and CR calculation from metal concentrations
- Calendar, lagged and strictly historical rolling features
- Correlation screening and VIF-based collinearity control
- Expanding-window TimeSeriesSplit evaluation rather than random splits
- CatBoost forecasting and walk-forward prediction export

The original study also tested deep-learning and stacked models. This public repository retains the compact, reproducible core and deliberately omits redundant experimental scripts and unpublished outputs.

## Run locally

```bash
pip install -r requirements.txt
python forecasting_pipeline.py --input data/private_input.xlsx --target PM2.5 --output outputs
```

Use `--target HI` or `--target CR` to calculate the corresponding risk index before forecasting. Review all risk-factor parameters against the intended population and formal assessment guideline before applied use.

## Data availability

Original environmental, meteorological, exposure and health-risk data are excluded because they may be unpublished or subject to use restrictions. Do not commit private inputs, generated metrics, figures, models, or manuscript drafts.
