# data/loader.py

import pandas as pd
from pathlib import Path


def load_table(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    try:
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(path, **kwargs)

        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path, **kwargs)

        if suffix == ".parquet":
            return pd.read_parquet(path, **kwargs)

        if suffix == ".json":
            return pd.read_json(path, **kwargs)

        raise ValueError(f"Unsupported file type: {suffix}")

    except Exception as e:
        raise ValueError(f"Failed to load {path.name}: {e}")
