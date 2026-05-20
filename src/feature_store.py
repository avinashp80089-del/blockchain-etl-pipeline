import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# feature groups served to downstream risk/fraud models
FEATURE_GROUPS = {
    "transaction_velocity": [
        "txn_count_1h", "txn_count_24h", "txn_count_7d",
        "amount_sum_1h", "amount_sum_24h", "amount_sum_7d",
        "amount_mean_1h", "amount_mean_24h", "amount_mean_7d",
        "unique_recipients_24h", "unique_chains_24h",
    ],
    "amount_statistics": [
        "amount_log", "amount_zscore", "amount_percentile",
        "fee_ratio", "total_cost_usd", "amount_momentum",
        "max_single_txn_7d", "amount_cv_7d",
    ],
    "behavioral": [
        "hour_of_day", "day_of_week", "is_weekend", "is_night",
        "preferred_chain", "preferred_token", "days_since_first_txn",
        "days_since_last_txn", "account_age_days",
    ],
    "risk_signals": [
        "velocity_x_amount", "cross_chain_activity", "high_value_flag",
        "rapid_succession_flag", "dormant_account_flag", "new_address_flag",
        "large_round_amount_flag", "off_hours_large_txn",
    ],
    "network": [
        "unique_counterparties_7d", "counterparty_risk_score_avg",
        "known_risky_address", "mixer_interaction", "exchange_volume_pct",
    ],
}


class FeatureStore:
    def __init__(self, store_path: str = "data/feature_store"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._registry: Dict[str, Dict] = self._load_registry()

    def register_feature_group(
        self,
        group_name: str,
        feature_definitions: List[str],
        entity_col: str = "from_address",
        event_time_col: str = "timestamp",
        description: str = "",
    ):
        self._registry[group_name] = {
            "features": feature_definitions,
            "entity_col": entity_col,
            "event_time_col": event_time_col,
            "description": description,
            "registered_at": datetime.utcnow().isoformat(),
            "record_count": 0,
        }
        self._save_registry()
        print(f"[FeatureStore] registered '{group_name}' ({len(feature_definitions)} features)")

    def ingest(self, group_name: str, df: pd.DataFrame):
        if group_name not in self._registry:
            raise ValueError(f"Feature group '{group_name}' not registered.")

        path = self.store_path / f"{group_name}.parquet"
        df["_ingested_at"] = datetime.utcnow().isoformat()

        if path.exists():
            df = pd.concat([pd.read_parquet(path), df], ignore_index=True)

        df.to_parquet(path, index=False)
        self._registry[group_name]["record_count"] = len(df)
        self._save_registry()
        print(f"[FeatureStore] ingested {len(df)} records → '{group_name}'")

    def get_features(
        self,
        group_name: str,
        entity_ids: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
        as_of: Optional[datetime] = None,
    ) -> pd.DataFrame:
        if group_name not in self._registry:
            raise ValueError(f"Feature group '{group_name}' not registered.")

        path = self.store_path / f"{group_name}.parquet"
        if not path.exists():
            return pd.DataFrame()

        df = pd.read_parquet(path)
        entity_col = self._registry[group_name]["entity_col"]
        event_time_col = self._registry[group_name]["event_time_col"]

        if as_of and event_time_col in df.columns:
            df = df[pd.to_datetime(df[event_time_col]) <= as_of]

        if entity_ids and entity_col in df.columns:
            df = df[df[entity_col].isin(entity_ids)]

        if feature_names:
            available = [f for f in feature_names if f in df.columns]
            df = df[[entity_col] + available]

        return df

    def join_feature_groups(self, group_names: List[str], entity_ids=None) -> pd.DataFrame:
        frames = {}
        for name in group_names:
            entity_col = self._registry.get(name, {}).get("entity_col", "from_address")
            df = self.get_features(name, entity_ids=entity_ids)
            if not df.empty:
                frames[name] = df.set_index(entity_col)

        if not frames:
            return pd.DataFrame()

        result = list(frames.values())[0]
        for df in list(frames.values())[1:]:
            result = result.join(df, how="outer")
        return result.reset_index()

    def describe(self) -> Dict[str, Any]:
        total = sum(len(v["features"]) for v in self._registry.values())
        return {
            "feature_groups": len(self._registry),
            "total_features": total,
            "groups": {k: {"features": len(v["features"]), "records": v["record_count"]} for k, v in self._registry.items()},
        }

    def _load_registry(self) -> Dict:
        p = self.store_path / "_registry.json"
        return json.load(open(p)) if p.exists() else {}

    def _save_registry(self):
        with open(self.store_path / "_registry.json", "w") as f:
            json.dump(self._registry, f, indent=2)


def build_training_feature_store(df: pd.DataFrame, store_path: str = "data/feature_store") -> FeatureStore:
    store = FeatureStore(store_path)
    for group_name, features in FEATURE_GROUPS.items():
        store.register_feature_group(group_name, features)
        available = [c for c in features if c in df.columns]
        if available and "from_address" in df.columns:
            store.ingest(group_name, df[["from_address", "timestamp"] + available].copy())
    print(f"\nFeature Store: {store.describe()}")
    return store
