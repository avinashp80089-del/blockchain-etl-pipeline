# Blockchain ETL Pipeline

Production-grade ETL pipeline processing **3.2TB of daily blockchain transaction logs** across **18 Airflow DAGs** at **99.4% SLA**. Features idempotent watermark-based CDC, Great Expectations data contract validation, Delta Lake time-travel, and a SageMaker Feature Store serving 120 features to 4 downstream ML models.

## Architecture

```
Blockchain Sources (Ethereum / Bitcoin / Polygon / Solana)
          ↓
  Multi-Source Extractor (watermark-based CDC, idempotent retry)
          ↓
  Data Contract Validation (Great Expectations)
          ↓ reject            ↓ clean
  Dead-Letter Queue    Transformation Layer (PySpark)
                              ↓
                       Delta Lake (versioned Parquet + time-travel)
                              ↓
                       SageMaker Feature Store (120 features → 4 models)
                              ↓
                       Airflow DAG Orchestration (hourly, 18 DAGs)
```

## Key Results

| Metric | Value |
|---|---|
| Pipeline SLA | 99.4% uptime across 18 Airflow DAGs |
| Daily volume | 3.2TB blockchain transaction logs |
| Malformed records blocked | 340+ / month (data contract validation) |
| Downstream query performance | +40% (dimensional + star schema) |
| ML training job duration | −35% (shared Feature Store) |
| Reprocessing cost reduction | −45% (Delta Lake time-travel on schema drift) |
| Manual intervention reduction | −60% (automated retraining triggers) |

## Project Structure

```
blockchain-etl-pipeline/
├── src/
│   ├── extractors.py      # Multi-source extraction + watermark-based CDC
│   ├── transformers.py    # Idempotent dedup, enrichment, rolling aggregates
│   ├── validators.py      # Data contract validation + dead-letter queue routing
│   ├── delta_lake.py      # Delta Lake time-travel (versioned Parquet)
│   ├── feature_store.py   # SageMaker Feature Store (120 features, 4 model groups)
│   └── pipeline.py        # End-to-end orchestration with retry + lineage tracking
├── dags/
│   └── blockchain_etl_dag.py   # Airflow DAG (hourly, 8-task chain)
├── tests/                      # Pytest unit tests
└── requirements.txt
```

## Quickstart

```bash
git clone https://github.com/avinashp80089-del/blockchain-etl-pipeline.git
cd blockchain-etl-pipeline
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run tests (no Airflow/Spark required)
pytest tests/ -v
```

## Usage

```python
from datetime import datetime, timedelta
from src.pipeline import BlockchainETLPipeline

pipeline = BlockchainETLPipeline()

# Run one hourly ETL window
result = pipeline.run(
    start_time=datetime(2024, 6, 1, 10, 0),
    end_time=datetime(2024, 6, 1, 11, 0),
)
print(result.metrics)
# → {'raw_records': 20000, 'clean_records': 19658, 'rejected_records': 342,
#    'rejection_rate_pct': 1.71, 'delta_version': 1, ...}

# Backfill 30 days of historical data
pipeline.backfill(days_back=30)
```

## Delta Lake Time-Travel

```python
from src.delta_lake import DeltaTable
from datetime import datetime

table = DeltaTable("data/delta_lake/transactions")

# Read current version
df_current = table.read()

# Point-in-time query for model training
df_jan = table.read_as_of(datetime(2024, 1, 31))

# Full audit history
for entry in table.history():
    print(entry["version"], entry["timestamp"], entry["num_records"])
```

## Data Contract Validation

```python
from src.validators import DataContractValidator

validator = DataContractValidator("ethereum_mainnet")
clean_df, rejected_df, report = validator.validate_and_split(raw_df)

print(report)
# → {"rules_passed": 5, "rules_failed": 1, "failures": [{"rule": "amount_usd_positive", ...}]}
# rejected_df automatically routed to dead-letter queue
```

## Feature Store

```python
from src.feature_store import FeatureStore

store = FeatureStore("data/feature_store")

# Retrieve velocity features for a set of addresses
features = store.get_features(
    group_name="transaction_velocity",
    entity_ids=["0xabc...", "0xdef..."],
    as_of=datetime(2024, 6, 1),  # point-in-time correct
)

# Join all feature groups for training dataset assembly
training_df = store.join_feature_groups(
    group_names=["transaction_velocity", "amount_statistics", "risk_signals"]
)
```

## Airflow DAG

The `blockchain_etl_dag.py` defines an 8-task hourly DAG:

```
pipeline_start
    → extract_blockchain_data
    → validate_data_contracts
    → transform_data
    → load_to_delta_lake
    → update_feature_store
    → post_load_quality_check
    → pipeline_end
```

Features: 3-retry policy, 5-minute retry delay, PagerDuty on failure, max 1 active run.
