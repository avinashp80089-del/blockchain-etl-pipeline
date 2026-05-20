import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from src.extractors import BlockchainExtractor, MultiSourceExtractor
from src.transformers import run_transformations
from src.validators import DataContractValidator, save_to_dead_letter_queue
from src.delta_lake import DeltaTable
from src.feature_store import FeatureStore, build_training_feature_store


DEFAULT_SOURCES = [
    {"source_name": "ethereum_mainnet", "records_per_hour": 8_000, "chains": ["ethereum"]},
    {"source_name": "bitcoin_mainnet",  "records_per_hour": 3_000, "chains": ["bitcoin"]},
    {"source_name": "polygon_matic",    "records_per_hour": 5_000, "chains": ["polygon"]},
    {"source_name": "solana_mainnet",   "records_per_hour": 4_000, "chains": ["solana"]},
]


class PipelineRun:
    def __init__(self, run_id: str, start: datetime, end: datetime):
        self.run_id = run_id
        self.start_time = start
        self.end_time = end
        self.status = "running"
        self.metrics: Dict[str, Any] = {}
        self.errors: List = []

    def to_dict(self):
        return {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "status": self.status,
            "metrics": self.metrics,
            "errors": self.errors,
        }


class BlockchainETLPipeline:
    def __init__(
        self,
        source_configs: list = None,
        delta_table_path: str = "data/delta_lake/transactions",
        feature_store_path: str = "data/feature_store",
        dead_letter_path: str = "data/dead_letter",
    ):
        self.extractor = MultiSourceExtractor(source_configs or DEFAULT_SOURCES)
        self.validator = DataContractValidator(source_name="blockchain_etl")
        self.delta = DeltaTable(delta_table_path)
        self.feature_store_path = feature_store_path
        self.dead_letter_path = dead_letter_path
        self._last_watermark: Optional[datetime] = None

    def run(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_retries: int = 3,
    ) -> PipelineRun:
        run_id = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
        end_time = end_time or datetime.utcnow()
        start_time = start_time or (self._last_watermark or end_time - timedelta(hours=1))

        run = PipelineRun(run_id, start_time, end_time)

        for attempt in range(1, max_retries + 1):
            try:
                run.metrics = self._execute(start_time, end_time)
                run.status = "success"
                self._last_watermark = end_time
                break
            except Exception as exc:
                run.errors.append({"attempt": attempt, "error": str(exc), "trace": traceback.format_exc()})
                print(f"[Pipeline] attempt {attempt}/{max_retries} failed: {exc}")
                if attempt == max_retries:
                    run.status = "failed"

        print(f"[Pipeline] {run_id}: {run.status.upper()}")
        return run

    def _execute(self, start: datetime, end: datetime) -> Dict[str, Any]:
        t0 = datetime.utcnow()

        raw = self.extractor.extract_all(start, end)
        print(f"[Extract] {len(raw)} records from {len(DEFAULT_SOURCES)} sources")

        clean, rejected, validation_report = self.validator.validate_and_split(raw)
        if not rejected.empty:
            save_to_dead_letter_queue(rejected, source="blockchain_etl", output_dir=self.dead_letter_path)

        transformed = run_transformations(clean)
        print(f"[Transform] {len(transformed)} records")

        delta_v = self.delta.write(
            transformed, mode="append",
            metadata={"run_start": start.isoformat(), "run_end": end.isoformat()},
        )

        fs = build_training_feature_store(transformed, self.feature_store_path)

        return {
            "raw_records": len(raw),
            "clean_records": len(clean),
            "rejected_records": len(rejected),
            "rejection_rate_pct": round(len(rejected) / max(len(raw), 1) * 100, 3),
            "delta_version": delta_v,
            "feature_store_summary": fs.describe(),
            "elapsed_seconds": round((datetime.utcnow() - t0).total_seconds(), 2),
            "validation_passed": validation_report["passed"],
        }

    def backfill(self, days_back: int = 30):
        print(f"[Backfill] {days_back}-day backfill starting...")
        now = datetime.utcnow()
        for day in range(days_back, 0, -1):
            start = now - timedelta(days=day)
            end = now - timedelta(days=day - 1)
            print(f"[Backfill] {start.date()} → {end.date()}")
            self.run(start_time=start, end_time=end)
        print("[Backfill] done.")
