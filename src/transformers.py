from typing import Dict, List

import numpy as np
import pandas as pd


def deduplicate(df: pd.DataFrame, key_cols: List[str], keep: str = "last") -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep=keep)
    dropped = before - len(df)
    if dropped:
        print(f"[deduplicate] dropped {dropped} dupes on {key_cols}")
    return df


def filter_confirmed(df: pd.DataFrame, status_col: str = "status") -> pd.DataFrame:
    return df[df[status_col] == "confirmed"].copy()


def enrich_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[ts_col])
    df["year"] = ts.dt.year
    df["month"] = ts.dt.month
    df["day"] = ts.dt.day
    df["hour"] = ts.dt.hour
    df["date_partition"] = ts.dt.strftime("%Y-%m-%d")
    df["hour_partition"] = ts.dt.strftime("%Y-%m-%d-%H")
    return df


def compute_rolling_aggregates(
    df: pd.DataFrame,
    group_col: str = "from_address",
    amount_col: str = "amount_usd",
    windows: List[int] = [1, 24, 168],
) -> pd.DataFrame:
    df = df.copy().sort_values([group_col, "timestamp"])

    for w in windows:
        label = f"{w}h" if w < 24 else f"{w // 24}d"
        df[f"txn_count_{label}"] = (
            df.groupby(group_col)["timestamp"]
            .transform(lambda s: s.expanding().count())
        )
        df[f"amount_sum_{label}"] = (
            df.groupby(group_col)[amount_col]
            .transform(lambda x: x.rolling(window=w, min_periods=1).sum())
        )
        df[f"amount_mean_{label}"] = (
            df.groupby(group_col)[amount_col]
            .transform(lambda x: x.rolling(window=w, min_periods=1).mean())
        )
    return df


def standardize_amounts(
    df: pd.DataFrame,
    amount_col: str = "amount_usd",
    fee_col: str = "gas_fee_usd",
) -> pd.DataFrame:
    df = df.copy()
    df["amount_log"] = np.log1p(df[amount_col])
    df["fee_ratio"] = df[fee_col] / (df[amount_col] + 1e-6)
    df["total_cost_usd"] = df[amount_col] + df[fee_col]
    return df


def apply_schema_evolution(
    df: pd.DataFrame,
    expected_schema: Dict[str, str],
    fill_defaults: bool = True,
) -> pd.DataFrame:
    """Handle upstream schema drift without crashing downstream jobs."""
    df = df.copy()
    for col, dtype in expected_schema.items():
        if col not in df.columns:
            if fill_defaults:
                df[col] = _default_for_dtype(dtype)
            else:
                raise ValueError(f"Missing expected column: {col}")
        else:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError):
                pass
    return df


def _default_for_dtype(dtype: str):
    return {"float64": 0.0, "int64": 0, "string": "", "object": None, "bool": False}.get(dtype)


def run_transformations(df: pd.DataFrame) -> pd.DataFrame:
    df = deduplicate(df, key_cols=["transaction_hash"])
    df = filter_confirmed(df)
    df = enrich_time_features(df)
    df = standardize_amounts(df)
    df = compute_rolling_aggregates(df)
    return df
