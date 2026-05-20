"""Unit tests for blockchain ETL pipeline."""
from datetime import datetime, timedelta
import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.extractors import BlockchainExtractor, MultiSourceExtractor
from src.transformers import deduplicate, filter_confirmed, enrich_time_features, standardize_amounts
from src.validators import DataContractValidator
from src.delta_lake import DeltaTable


# ── Extractors ────────────────────────────────────────────────────────────────

def test_extractor_returns_dataframe():
    extractor = BlockchainExtractor({"source_name": "test", "records_per_hour": 100})
    end = datetime.utcnow()
    start = end - timedelta(hours=1)
    df = extractor.extract_batch(start, end)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_extractor_idempotent():
    """Same time window → same row count (deterministic seed)."""
    extractor = BlockchainExtractor({"source_name": "test", "records_per_hour": 200})
    end = datetime(2024, 6, 1, 12, 0, 0)
    start = end - timedelta(hours=1)
    df1 = extractor.extract_batch(start, end)
    df2 = extractor.extract_batch(start, end)
    assert len(df1) == len(df2)


def test_watermark_updates_after_extract():
    extractor = BlockchainExtractor({"source_name": "test", "records_per_hour": 50})
    assert extractor.watermark is None
    end = datetime.utcnow()
    extractor.extract_batch(end - timedelta(hours=1), end)
    assert extractor.watermark is not None


def test_multi_source_extractor():
    configs = [
        {"source_name": "src1", "records_per_hour": 100, "chains": ["ethereum"]},
        {"source_name": "src2", "records_per_hour": 100, "chains": ["bitcoin"]},
    ]
    extractor = MultiSourceExtractor(configs)
    end = datetime.utcnow()
    df = extractor.extract_all(end - timedelta(hours=1), end)
    assert len(df) > 0


# ── Transformers ──────────────────────────────────────────────────────────────

def _make_df() -> pd.DataFrame:
    return pd.DataFrame({
        "transaction_hash": ["0xaaa", "0xbbb", "0xaaa", "0xccc"],
        "timestamp": pd.date_range("2024-01-01", periods=4, freq="h"),
        "amount_usd": [100.0, 200.0, 100.0, 50.0],
        "gas_fee_usd": [1.0, 2.0, 1.0, 0.5],
        "status": ["confirmed", "failed", "confirmed", "confirmed"],
        "from_address": ["0x1", "0x2", "0x1", "0x3"],
    })


def test_deduplicate():
    df = _make_df()
    result = deduplicate(df, key_cols=["transaction_hash"])
    assert len(result) == 3
    assert result["transaction_hash"].duplicated().sum() == 0


def test_filter_confirmed():
    df = _make_df()
    result = filter_confirmed(df)
    assert (result["status"] == "confirmed").all()
    assert len(result) == 3


def test_enrich_time_features():
    df = _make_df()
    result = enrich_time_features(df)
    for col in ["year", "month", "day", "hour", "date_partition"]:
        assert col in result.columns


def test_standardize_amounts():
    df = _make_df()
    result = standardize_amounts(df)
    assert "amount_log" in result.columns
    assert "fee_ratio" in result.columns
    assert (result["amount_log"] >= 0).all()


# ── Validators ────────────────────────────────────────────────────────────────

def test_validator_passes_clean_data():
    df = pd.DataFrame({
        "transaction_hash": ["0xaaa", "0xbbb"],
        "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        "chain": ["ethereum", "bitcoin"],
        "amount_usd": [100.0, 200.0],
        "status": ["confirmed", "failed"],
    })
    validator = DataContractValidator("test_source")
    report = validator.validate(df)
    assert report["passed"] is True


def test_validator_catches_negative_amounts():
    df = pd.DataFrame({
        "transaction_hash": ["0xaaa"],
        "timestamp": [datetime(2024, 1, 1)],
        "chain": ["ethereum"],
        "amount_usd": [-50.0],
        "status": ["confirmed"],
    })
    validator = DataContractValidator("test_source")
    report = validator.validate(df)
    assert report["passed"] is False
    assert any(f["rule"] == "amount_usd_positive" for f in report["failures"])


def test_validator_catches_duplicate_hashes():
    df = pd.DataFrame({
        "transaction_hash": ["0xaaa", "0xaaa"],
        "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        "chain": ["ethereum", "ethereum"],
        "amount_usd": [100.0, 200.0],
        "status": ["confirmed", "confirmed"],
    })
    validator = DataContractValidator("test_source")
    report = validator.validate(df)
    assert report["passed"] is False


def test_validator_split():
    df = pd.DataFrame({
        "transaction_hash": ["0xaaa", "0xbbb"],
        "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        "chain": ["ethereum", "bitcoin"],
        "amount_usd": [100.0, -5.0],
        "status": ["confirmed", "confirmed"],
    })
    validator = DataContractValidator("test_source")
    clean, rejected, _ = validator.validate_and_split(df)
    assert len(rejected) == 1
    assert len(clean) == 1


# ── Delta Lake ────────────────────────────────────────────────────────────────

_parquet_available = pytest.mark.skipif(
    True,
    reason="pyarrow or fastparquet required — install with: pip install pyarrow"
)
try:
    import pyarrow  # noqa: F401
    _parquet_available = lambda f: f  # no-op decorator when available
except ImportError:
    pass


@_parquet_available
def test_delta_write_and_read():
    with tempfile.TemporaryDirectory() as tmpdir:
        table = DeltaTable(tmpdir + "/test_table")
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10.0, 20.0, 30.0]})
        version = table.write(df, mode="append")
        assert version == 1
        result = table.read()
        assert len(result) == 3


@_parquet_available
def test_delta_time_travel():
    with tempfile.TemporaryDirectory() as tmpdir:
        table = DeltaTable(tmpdir + "/test_table")
        df1 = pd.DataFrame({"id": [1], "value": [10.0]})
        df2 = pd.DataFrame({"id": [2], "value": [20.0]})
        table.write(df1)
        table.write(df2)

        v1_data = table.read(version=1)
        v2_data = table.read(version=2)
        assert len(v1_data) == 1
        assert len(v2_data) == 2


@_parquet_available
def test_delta_history():
    with tempfile.TemporaryDirectory() as tmpdir:
        table = DeltaTable(tmpdir + "/test_table")
        df = pd.DataFrame({"id": [1]})
        table.write(df)
        table.write(df)
        history = table.history()
        assert len(history) == 2
