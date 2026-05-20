import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd


class ValidationResult:
    def __init__(self, rule: str, passed: bool, details: Dict[str, Any]):
        self.rule = rule
        self.passed = passed
        self.details = details

    def to_dict(self):
        return {"rule": self.rule, "passed": self.passed, "details": self.details}


class DataContractValidator:
    """Enforce data contracts at ingestion — bad rows go to DLQ, not the warehouse."""

    def __init__(self, source_name: str):
        self.source_name = source_name
        self._rules: List[Callable] = []
        self._register_defaults()

    def _register_defaults(self):
        for rule in [
            self._check_no_nulls_in_keys,
            self._check_amount_positive,
            self._check_timestamp_range,
            self._check_status_values,
            self._check_hash_uniqueness,
            self._check_chain_values,
        ]:
            self._rules.append(rule)

    def add_rule(self, fn: Callable):
        self._rules.append(fn)

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        results = [r(df) for r in self._rules]
        failed = [r for r in results if not r.passed]

        report = {
            "source": self.source_name,
            "timestamp": datetime.utcnow().isoformat(),
            "total_records": len(df),
            "rules_passed": sum(r.passed for r in results),
            "rules_failed": len(failed),
            "passed": len(failed) == 0,
            "failures": [r.to_dict() for r in failed],
        }

        if failed:
            print(f"[VALIDATION FAILED] {self.source_name}: {[r.rule for r in failed]}")
        else:
            print(f"[VALIDATION PASSED] {self.source_name}: {len(df)} records clean")

        return report

    def validate_and_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        report = self.validate(df)
        bad = self._row_level_mask(df)
        return df[~bad].copy(), df[bad].copy(), report

    def _row_level_mask(self, df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(False, index=df.index)
        if "amount_usd" in df.columns:
            mask |= df["amount_usd"].isna() | (df["amount_usd"] <= 0)
        if "transaction_hash" in df.columns:
            mask |= df["transaction_hash"].isna()
        if "timestamp" in df.columns:
            mask |= pd.to_datetime(df["timestamp"], errors="coerce").isna()
        return mask

    # ── individual rules ──────────────────────────────────────────────────────

    def _check_no_nulls_in_keys(self, df):
        key_cols = ["transaction_hash", "timestamp", "chain"]
        null_counts = {c: int(df[c].isna().sum()) for c in key_cols if c in df.columns}
        return ValidationResult(
            "no_nulls_in_key_columns",
            passed=sum(null_counts.values()) == 0,
            details={"null_counts": null_counts},
        )

    def _check_amount_positive(self, df):
        if "amount_usd" not in df.columns:
            return ValidationResult("amount_positive", False, {"error": "column missing"})
        invalid = int((df["amount_usd"] <= 0).sum())
        return ValidationResult(
            "amount_usd_positive",
            passed=invalid == 0,
            details={"invalid_count": invalid, "pct": round(invalid / len(df) * 100, 3)},
        )

    def _check_timestamp_range(self, df):
        if "timestamp" not in df.columns:
            return ValidationResult("timestamp_range", False, {"error": "column missing"})
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        now = datetime.utcnow()
        return ValidationResult(
            "timestamp_in_valid_range",
            passed=(ts > now).sum() == 0 and (ts < pd.Timestamp("2015-01-01")).sum() == 0,
            details={"future": int((ts > now).sum()), "too_old": int((ts < pd.Timestamp("2015-01-01")).sum())},
        )

    def _check_status_values(self, df):
        allowed = {"confirmed", "failed", "pending"}
        if "status" not in df.columns:
            return ValidationResult("status_values", True, {"note": "column not present"})
        invalid = df[~df["status"].isin(allowed)]
        return ValidationResult(
            "status_valid_enum",
            passed=len(invalid) == 0,
            details={"invalid_count": len(invalid)},
        )

    def _check_hash_uniqueness(self, df):
        if "transaction_hash" not in df.columns:
            return ValidationResult("hash_uniqueness", False, {"error": "column missing"})
        dupes = int(df["transaction_hash"].duplicated().sum())
        return ValidationResult("transaction_hash_unique", passed=dupes == 0, details={"duplicate_count": dupes})

    def _check_chain_values(self, df):
        allowed = {"ethereum", "bitcoin", "polygon", "solana", "avalanche", "arbitrum"}
        if "chain" not in df.columns:
            return ValidationResult("chain_values", True, {"note": "column not present"})
        invalid = df[~df["chain"].isin(allowed)]
        return ValidationResult("chain_valid_enum", passed=len(invalid) == 0, details={"invalid_count": len(invalid)})


def save_to_dead_letter_queue(rejected: pd.DataFrame, source: str, output_dir: str = "data/dead_letter") -> str:
    import os
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = f"{output_dir}/{source}_{ts}.parquet"
    rejected.to_parquet(path, index=False)
    print(f"[DLQ] {len(rejected)} records → {path}")
    return path
