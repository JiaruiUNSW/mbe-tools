from __future__ import annotations
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union, SupportsFloat, SupportsIndex, SupportsInt, cast
import itertools

Number = float
NumericLike = Union[SupportsFloat, SupportsIndex, int, float, str]


def _get_attr(obj: Mapping[str, object], key: str):
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key)


IndexLike = Union[int, Iterable[SupportsInt]]


def _normalize_indices(indices: IndexLike) -> Tuple[int, ...]:
    """Normalize subset indices to a sorted tuple.

    Accepts scalar ints for convenience (e.g., subset_indices=1), and raises
    a ValueError if the input cannot be coerced.
    """
    if isinstance(indices, int):
        return (int(indices),)
    try:
        return tuple(sorted(int(i) for i in indices))
    except TypeError as exc:  # not iterable
        raise ValueError(f"subset_indices must be an iterable of ints; got {indices!r}") from exc


def build_energy_map(records: Iterable[Mapping[str, object]]) -> Dict[Tuple[int, ...], Number]:
    """Build a mapping of subset -> energy from parsed records.

    Records missing subset_indices or energy_hartree are skipped.
    """
    mapping: Dict[Tuple[int, ...], Number] = {}
    for idx, rec in enumerate(records):
        subset_indices = _get_attr(rec, "subset_indices")
        energy = _get_attr(rec, "energy_hartree")
        if subset_indices is None or energy is None:
            raise ValueError(
                f"Record {idx}: subset_indices and energy_hartree are required; got subset_indices={subset_indices!r}, energy_hartree={energy!r}"
            )
        key = _normalize_indices(cast(IndexLike, subset_indices))
        try:
            energy_val = float(cast(NumericLike, energy))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"energy_hartree must be numeric; got {energy!r}") from exc
        mapping[key] = energy_val
    return mapping


def _proper_subsets(idx: Tuple[int, ...]):
    for k in range(1, len(idx)):
        for sub in itertools.combinations(idx, k):
            yield sub


def compute_contributions(energy_map: Mapping[Tuple[int, ...], Number], max_order: Optional[int] = None):
    """Inclusion–exclusion contributions per subset.

    Returns (contribs, missing_subsets) where contribs maps subset -> ΔE(S).
    """
    contribs: Dict[Tuple[int, ...], Number] = {}
    missing: List[Tuple[int, ...]] = []

    sizes = sorted({len(k) for k in energy_map})
    target_orders = [k for k in sizes if max_order is None or k <= max_order]

    for k in target_orders:
        subsets = [s for s in energy_map if len(s) == k]
        for subset in subsets:
            subtotal = energy_map[subset]
            for sub in _proper_subsets(subset):
                if sub in contribs:
                    subtotal -= contribs[sub]
                else:
                    # missing lower-order contribution
                    missing.append(sub)
            contribs[subset] = subtotal
    return contribs, missing


def assemble_mbe_energy(records: Iterable[Mapping[str, object]], max_order: Optional[int] = None):
    """Assemble MBE(k) energies via inclusion–exclusion.

    Returns a dict with:
      - contributions: ΔE for each subset
      - order_totals: cumulative MBE(k) energy
      - missing_subsets: list of missing lower-order subsets encountered
    """
    energy_map = build_energy_map(records)
    contribs, missing = compute_contributions(energy_map, max_order=max_order)

    order_totals: Dict[int, Number] = {}
    if contribs:
        order_limit = max_order or max(len(s) for s in contribs)
        for subset, val in contribs.items():
            size = len(subset)
            for k in range(size, order_limit + 1):
                order_totals[k] = order_totals.get(k, 0.0) + val

    return {
        "contributions": contribs,
        "order_totals": order_totals,
        "missing_subsets": missing,
    }


def order_totals_as_rows(order_totals: Mapping[int, Number]):
    rows = []
    for order in sorted(order_totals):
        rows.append({"order": order, "mbe_energy_hartree": order_totals[order]})
    return rows
