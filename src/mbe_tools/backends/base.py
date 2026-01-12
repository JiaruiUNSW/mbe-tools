from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional
from ..utils import Atom

@dataclass(frozen=True)
class BackendFormat:
    """Defines how to represent atoms and ghost atoms for a given quantum chemistry engine."""
    name: str

    def format_atom(self, atom: Atom, ghost: bool = False) -> str:
        raise NotImplementedError

def get_backend(name: str) -> BackendFormat:
    name_l = name.lower()
    if name_l in ("qchem", "q-chem"):
        from .qchem import QChemBackend
        return QChemBackend()
    if name_l in ("orca",):
        from .orca import OrcaBackend
        return OrcaBackend()
    raise ValueError(f"Unknown backend: {name}")
