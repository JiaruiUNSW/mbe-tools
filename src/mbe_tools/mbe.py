from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple, Optional
import itertools
import hashlib

from .utils import Atom
from .cluster import Fragment, FragmentRecord
from .backends.base import get_backend

@dataclass(frozen=True)
class MBEParams:
    max_order: int = 2
    orders: Optional[Sequence[int]] = None
    cp_correction: bool = True
    backend: str = "qchem"  # 'qchem' or 'orca'
    name_prefix: Optional[str] = None  # override filename prefix (else backend)
    charge: int = 0
    multiplicity: int = 1
    scheme: str = "mbe"
    thresh: Optional[float] = None
    tole: Optional[float] = None
    scf_convergence: Optional[str] = None

def _atoms_of(fragment: Sequence[Atom] | FragmentRecord) -> List[Atom]:
    if isinstance(fragment, FragmentRecord):
        return list(fragment.atoms)
    return list(fragment)


def _all_atoms(fragments: Sequence[Fragment | FragmentRecord]) -> List[Atom]:
    out: List[Atom] = []
    for f in fragments:
        out.extend(_atoms_of(f))
    return out

def generate_subsets_xyz(
    fragments: Sequence[Fragment | FragmentRecord],
    params: MBEParams,
    orders: Optional[Sequence[int]] = None
) -> Iterator[Tuple[str, Tuple[int, ...], str]]:
    """Generate subset geometries up to params.max_order.

    Yields:
      (job_id, subset_indices, geometry_text)

    geometry_text is backend-specific coordinate block (NOT full input file),
    where atoms not in subset become ghosts if cp_correction=True.
    """
    if params.max_order < 1:
        raise ValueError("max_order must be >= 1")
    backend = get_backend(params.backend)

    frag_indices = list(range(len(fragments)))
    raw_orders = orders if orders is not None else params.orders
    if raw_orders is None:
        target_orders = list(range(1, params.max_order + 1))
    else:
        target_orders = sorted({int(k) for k in raw_orders if k is not None})
        if not target_orders:
            raise ValueError("orders must not be empty when provided")

    for k in target_orders:
        if k < 1:
            raise ValueError("orders must be >= 1")

    # Pre-flatten for speed
    frag_atoms = [_atoms_of(f) for f in fragments]

    for k in target_orders:
        for subset in itertools.combinations(frag_indices, k):
            # Build lines: real atoms for subset, ghost for others (if cp)
            lines: List[str] = []
            subset_set = set(subset)
            for fi in frag_indices:
                is_ghost = params.cp_correction and (fi not in subset_set)
                for atom in frag_atoms[fi]:
                    lines.append(backend.format_atom(atom, ghost=is_ghost))

            # deterministic job id, include human-readable indices (1-based); hash not needed
            subset_str = ".".join(str(i + 1) for i in subset)
            prefix = params.name_prefix or params.backend
            job_id = f"{prefix}_k{k}_{subset_str}"
            geom = "\n".join(lines)
            yield job_id, subset, geom

def qchem_molecule_block(geom_block: str, charge: int = 0, multiplicity: int = 1) -> str:
    return "$molecule\n" + f"{charge} {multiplicity}\n" + geom_block + "\n$end\n"

def orca_xyz_block(geom_block: str) -> str:
    # ORCA commonly uses '* xyz charge mult' blocks.
    return "* xyz 0 1\n" + geom_block + "\n*\n"
