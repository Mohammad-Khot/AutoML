import pandas as pd
from pathlib import Path
from typing import Any


def load_table(source: str | Path, **read_kwargs: Any) -> pd.DataFrame:
    """
    Load a tabular dataset from a file into a pandas DataFrame.

    Supports CSV, TXT, Excel (XLS/XLSX), Parquet, and JSON formats.
    Additional keyword arguments are passed directly to the corresponding
    pandas reader function.

    Args:
        source (str | Path): Path to the input file.
        **read_kwargs (Any): Additional arguments for pandas read functions.

    Returns:
        pd.DataFrame: Loaded dataset as a DataFrame.

    Raises:
        ValueError: If the file type is unsupported or loading fails.
    """
    file_path: Path = Path(source)
    file_ext: str = file_path.suffix.lower()

    try:
        if file_ext in {".csv", ".txt"}:
            return pd.read_csv(file_path, **read_kwargs)

        if file_ext in {".xlsx", ".xls"}:
            return pd.read_excel(file_path, **read_kwargs)

        if file_ext == ".parquet":
            return pd.read_parquet(file_path, **read_kwargs)

        if file_ext == ".json":
            return pd.read_json(file_path, **read_kwargs)

        raise ValueError(f"Unsupported file type: {file_path}")

    except Exception as err:
        raise ValueError(f"Failed to load {file_path.name}: {err}") from err
