import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class DeltaTable:
    """
    Local Delta Lake using versioned parquet + JSON transaction log.
    Swap _data_dir writes for S3 paths in prod.
    """

    def __init__(self, table_path: str):
        self.table_path = Path(table_path)
        self._data = self.table_path / "_data"
        self._log = self.table_path / "_delta_log"
        self._data.mkdir(parents=True, exist_ok=True)
        self._log.mkdir(parents=True, exist_ok=True)
        self._version = self._current_version()

    @property
    def version(self):
        return self._version

    def write(
        self,
        df: pd.DataFrame,
        mode: str = "append",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        new_v = self._version + 1
        vstr = f"{new_v:010d}"

        fpath = self._data / f"part-{vstr}.parquet"
        df.to_parquet(fpath, index=False)

        entry = {
            "version": new_v,
            "timestamp": datetime.utcnow().isoformat(),
            "operation": mode.upper(),
            "file": str(fpath),
            "num_records": len(df),
            "schema": df.dtypes.astype(str).to_dict(),
            "metadata": metadata or {},
        }

        if mode == "overwrite":
            prev = list(self._data.glob("*.parquet"))
            entry["removed_files"] = [str(f) for f in prev if f != fpath]

        with open(self._log / f"{vstr}.json", "w") as f:
            json.dump(entry, f, indent=2)

        self._version = new_v
        print(f"[DeltaTable] v{new_v}: wrote {len(df)} records ({mode})")
        return new_v

    def read(self, version: Optional[int] = None) -> pd.DataFrame:
        target = version if version is not None else self._version

        if target > self._version:
            raise ValueError(f"v{target} doesn't exist (current: v{self._version})")

        entries = self._load_log(up_to=target)
        active = self._active_files(entries)

        if not active:
            return pd.DataFrame()

        frames = [pd.read_parquet(f) for f in active if Path(f).exists()]
        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        print(f"[DeltaTable] read v{target}: {len(df)} rows from {len(active)} files")
        return df

    def read_as_of(self, ts: datetime) -> pd.DataFrame:
        entries = self._load_log()
        target_v = 0
        for e in entries:
            if datetime.fromisoformat(e["timestamp"]) <= ts:
                target_v = e["version"]
        return self.read(version=target_v)

    def history(self) -> List[Dict[str, Any]]:
        return self._load_log()

    def vacuum(self, retain_versions: int = 10):
        logs = self._load_log()
        if len(logs) <= retain_versions:
            return
        cutoff = len(logs) - retain_versions
        old_files = {e.get("file") for e in logs[:cutoff]}
        keep_files = {e.get("file") for e in logs[cutoff:]}
        for fpath in old_files - keep_files:
            p = Path(fpath)
            if p.exists():
                p.unlink()
                print(f"[vacuum] removed {p.name}")

    def _load_log(self, up_to: Optional[int] = None) -> List[Dict]:
        entries = []
        for lf in sorted(self._log.glob("*.json")):
            with open(lf) as f:
                e = json.load(f)
            if up_to is None or e["version"] <= up_to:
                entries.append(e)
        return entries

    def _active_files(self, entries: List[Dict]) -> List[str]:
        active, removed = set(), set()
        for e in entries:
            active.add(e["file"])
            removed.update(e.get("removed_files", []))
        return list(active - removed)

    def _current_version(self) -> int:
        logs = sorted(self._log.glob("*.json"))
        if not logs:
            return 0
        with open(logs[-1]) as f:
            return json.load(f).get("version", 0)
