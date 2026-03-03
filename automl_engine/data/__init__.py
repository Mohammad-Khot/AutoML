# data/__init.py

from .leakage import run_leakage_checks
from .loader import load_table
from .schema import infer_target, infer_task

__all__ = [
    "run_leakage_checks",
    "load_table",
    "infer_target",
    "infer_task",
]
