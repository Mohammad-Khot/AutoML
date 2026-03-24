# data/__init.py

from .leakage import apply_leakage_policy
from .loader import load_table
from .schema import infer_target, infer_task

__all__ = [
    "apply_leakage_policy",
    "load_table",
    "infer_target",
    "infer_task",
]
