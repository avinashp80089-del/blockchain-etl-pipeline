import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd


SUPPORTED_CHAINS = ["ethereum", "bitcoin", "polygon", "solana", "avalanche", "arbitrum"]


class BlockchainExtractor:
    def __init__(self, source_config: Dict[str, Any]):
        self.config = source_config
        self._watermark: Optional[datetime] = None

    @property
    def watermark(self):
        return self._watermark

    def extract_batch(self, start: datetime, end: datetime) -> pd.DataFrame:
        df = _make_synthetic_data(start, end, self.config)
        self._watermark = end
        return df

    def extract_incremental(
        self, last_watermark: Optional[datetime] = None, window_minutes: int = 60
    ) -> pd.DataFrame:
        # idempotent — re-extracting same watermark produces same result
        start = last_watermark or (datetime.utcnow() - timedelta(minutes=window_minutes))
        end = datetime.utcnow()
        df = self.extract_batch(start, end)
        self._watermark = end
        return df

    def stream(self, batch_size: int = 1_000) -> Iterator[pd.DataFrame]:
        now = datetime.utcnow()
        df = self.extract_batch(now - timedelta(hours=1), now)
        for i in range(0, len(df), batch_size):
            yield df.iloc[i: i + batch_size].copy()


def _make_synthetic_data(start: datetime, end: datetime, config: Dict) -> pd.DataFrame:
    seed = int(hashlib.md5(f"{start}{end}".encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed % (2 ** 31))

    n_per_hour = config.get("records_per_hour", 5_000)
    hours = max(1, int((end - start).total_seconds() / 3600))
    n = n_per_hour * hours

    chains = config.get("chains", SUPPORTED_CHAINS[:4])
    timestamps = pd.date_range(start, end, periods=n)

    return pd.DataFrame({
        "transaction_hash": [f"0x{rng.bytes(32).hex()[:64]}" for _ in range(n)],
        "block_number": rng.randint(18_000_000, 19_000_000, n),
        "timestamp": timestamps,
        "chain": rng.choice(chains, n),
        "from_address": [f"0x{rng.bytes(20).hex()[:40]}" for _ in range(n)],
        "to_address": [f"0x{rng.bytes(20).hex()[:40]}" for _ in range(n)],
        "amount_usd": np.abs(rng.lognormal(5.0, 2.0, n)),
        "gas_fee_usd": np.abs(rng.lognormal(1.5, 0.8, n)),
        "token_symbol": rng.choice(["ETH", "BTC", "USDT", "USDC", "MATIC"], n, p=[0.3, 0.2, 0.25, 0.15, 0.1]),
        "tx_type": rng.choice(["transfer", "swap", "mint", "burn", "stake"], n, p=[0.5, 0.25, 0.1, 0.05, 0.1]),
        "status": rng.choice(["confirmed", "failed", "pending"], n, p=[0.95, 0.04, 0.01]),
        "ingested_at": datetime.utcnow().isoformat(),
        "_source": config.get("source_name", "blockchain_api"),
    })


class MultiSourceExtractor:
    SCHEMA = {
        "transaction_hash": "string",
        "timestamp": "datetime64[ns]",
        "chain": "string",
        "amount_usd": "float64",
        "gas_fee_usd": "float64",
        "token_symbol": "string",
        "tx_type": "string",
        "status": "string",
    }

    def __init__(self, source_configs: List[Dict[str, Any]]):
        self.extractors = [BlockchainExtractor(cfg) for cfg in source_configs]

    def extract_all(self, start: datetime, end: datetime) -> pd.DataFrame:
        frames = [e.extract_batch(start, end) for e in self.extractors]
        combined = pd.concat(frames, ignore_index=True)
        return self._normalize(combined)

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, dtype in self.SCHEMA.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except (ValueError, TypeError):
                    pass
        return df[[c for c in self.SCHEMA if c in df.columns]]
