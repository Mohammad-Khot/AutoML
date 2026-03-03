# data/loader.py

import pandas as pd
from pathlib import Path
from typing import Any


def load_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """
    Load a tabular dataset from a file into a pandas DataFrame.

    Supported file formats:
        - .csv, .txt  -> pd.read_csv
        - .xlsx, .xls -> pd.read_excel
        - .parquet    -> pd.read_parquet
        - .json       -> pd.read_json

    Args:
        path (str | Path): Path to the dataset file.
        **kwargs (Any): Additional keyword arguments passed to the underlying
                        pandas reader function.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.

    Raises:
        ValueError: If the file type is unsupported or loading fails.
    """
    file_path: Path = Path(path)
    suffix: str = file_path.suffix.lower()

    try:
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(file_path, **kwargs)

        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(file_path, **kwargs)

        if suffix == ".parquet":
            return pd.read_parquet(file_path, **kwargs)

        if suffix == ".json":
            return pd.read_json(file_path, **kwargs)

        raise ValueError(f"Unsupported file type: {suffix}")

    except Exception as e:
        raise ValueError(f"Failed to load {file_path.name}: {e}") from e
