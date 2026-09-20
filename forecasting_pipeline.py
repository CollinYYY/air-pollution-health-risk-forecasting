"""Compact, leakage-aware daily forecasting workflow (portfolio edition)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.outliers_influence import variance_inflation_factor


def load_daily_data(path: Path, date_column: str) -> pd.DataFrame:
    data = pd.read_excel(path)
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data = data.dropna(subset=[date_column]).set_index(date_column).sort_index()
    data = data.loc[:, ~data.columns.str.contains(r"^Unnamed|站点", regex=True)]
    if data.index.has_duplicates:
        raise ValueError("Aggregate duplicate dates before forecasting.")
    return data


def calculate_health_risks(data: pd.DataFrame) -> pd.DataFrame:
    """Illustrative daily HI/CR calculation; review parameters before applied use."""
    metals = {m: next((c for c in data if m.lower() in c.lower()), None)
              for m in ("As", "Pb", "Cu", "Ni", "Co", "Cr", "Cd", "V")}
    if not any(metals.values()):
        raise ValueError("Expected metal columns were not found.")
    x = data.rename(columns={c: m for m, c in metals.items() if c})
    rfc = {"As": 1.50e-5, "Pb": 3.52e-3, "Cu": 1.80e-4, "Ni": 1.40e-5, "Co": 6.00e-6}
    iur = {"Cr": 1.20e-2, "Pb": 1.20e-5, "Cd": 1.80e-3, "V": 8.30e-3, "As": 4.30e-3}
    hi, cr = pd.Series(0.0, index=x.index), pd.Series(0.0, index=x.index)
    for m, value in rfc.items():
        if m in x: hi += x[m] * 1.1 * 24 / (70 * value * 1000)
    for m, value in iur.items():
        if m in x: cr += x[m] * 20 / 70 * value
    return pd.DataFrame({"HI": hi, "CR": cr})


def make_historical_features(data: pd.DataFrame, lags: int = 7) -> pd.DataFrame:
    """All lagged/rolling features only use information available before day t."""
    out = data.copy()
    out["weekday"] = out.index.weekday
    out["month_sin"] = np.sin(2 * np.pi * out.index.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * out.index.month / 12)
    for col in data.select_dtypes("number"):
        for lag in range(1, lags + 1): out[f"{col}_lag_{lag}"] = out[col].shift(lag)
        for window in (7, 14, 30): out[f"{col}_mean_{window}"] = out[col].rolling(window).mean().shift(1)
    return out


def select_features(data: pd.DataFrame, target: str, top_k: int = 15, vif_limit: float = 10) -> list[str]:
    numeric = data.select_dtypes("number").replace([np.inf, -np.inf], np.nan)
    candidates = numeric.drop(columns=target).dropna(axis=1, how="all")
    selected = candidates.corrwith(numeric[target]).abs().nlargest(top_k).index.tolist()
    while len(selected) > 1:
        matrix = candidates[selected].dropna()
        vifs = pd.Series([variance_inflation_factor(matrix.values, i) for i in range(matrix.shape[1])], index=selected)
        if vifs.max() <= vif_limit: break
        selected.remove(vifs.idxmax())
    return selected


def walk_forward_evaluate(data: pd.DataFrame, features: list[str], target: str, splits: int = 5):
    model_data = data[features + [target]].dropna()
    predictions, metrics = [], []
    for fold, (train_i, test_i) in enumerate(TimeSeriesSplit(n_splits=splits).split(model_data), 1):
        train, test = model_data.iloc[train_i], model_data.iloc[test_i]
        model = CatBoostRegressor(loss_function="RMSE", iterations=500, depth=6,
                                  learning_rate=0.05, random_seed=42, verbose=False)
        model.fit(train[features], train[target])
        pred = model.predict(test[features])
        predictions.append(pd.DataFrame({"date": test.index, "fold": fold, "actual": test[target], "prediction": pred}))
        metrics.append({"fold": fold, "RMSE": np.sqrt(mean_squared_error(test[target], pred)),
                        "MAE": mean_absolute_error(test[target], pred), "R2": r2_score(test[target], pred)})
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Authorised private Excel input; never commit it.")
    parser.add_argument("--target", required=True, help="Numeric column, or HI/CR to calculate before forecasting.")
    parser.add_argument("--date-column", default="日期")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    data = load_daily_data(args.input, args.date_column)
    if args.target in {"HI", "CR"}: data = data.join(calculate_health_risks(data))
    if args.target not in data: raise KeyError(f"Unknown target: {args.target}")
    featured = make_historical_features(data)
    features = select_features(featured, args.target)
    predictions, metrics = walk_forward_evaluate(featured, features, args.target)
    args.output.mkdir(exist_ok=True)
    predictions.to_csv(args.output / "walk_forward_predictions.csv", index=False)
    metrics.to_csv(args.output / "walk_forward_metrics.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(args.output / "selected_features.csv", index=False)


if __name__ == "__main__":
    main()
