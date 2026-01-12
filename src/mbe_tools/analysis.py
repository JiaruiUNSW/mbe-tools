from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterable, Tuple
import json
import math

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows

def to_dataframe(records: List[Dict[str, Any]]):
    if pd is None:
        raise RuntimeError("pandas is required. Install with: pip install mbe-tools[analysis]")
    return pd.DataFrame(records)

def summarize_by_order(df):
    # expects subset_size, energy_hartree, cpu_seconds
    g = df.groupby("subset_size", dropna=False).agg(
        n=("job_id", "count"),
        energy_min=("energy_hartree", "min"),
        energy_max=("energy_hartree", "max"),
        cpu_total=("cpu_seconds", "sum"),
        cpu_mean=("cpu_seconds", "mean"),
    ).reset_index()
    return g

def compute_delta_energy(df, reference_order: int = 1):
    # ΔE_k = mean(E_k) - mean(E_ref) (simple; you may want true inclusion-exclusion later)
    ref = df[df["subset_size"] == reference_order]["energy_hartree"].mean()
    df = df.copy()
    df["delta_energy_hartree_vs_ref"] = df["energy_hartree"] - ref
    return df


def strict_mbe_orders(records: List[Dict[str, Any]], max_order: Optional[int] = None):
    """Return inclusion–exclusion MBE(k) energies as a list of dicts."""
    from .mbe_math import assemble_mbe_energy, order_totals_as_rows

    result = assemble_mbe_energy(records, max_order=max_order)
    rows = order_totals_as_rows(result["order_totals"])
    return rows, result.get("missing_subsets", [])
