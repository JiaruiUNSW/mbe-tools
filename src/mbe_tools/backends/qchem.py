from __future__ import annotations
from dataclasses import dataclass
from ..utils import Atom
from .base import BackendFormat

@dataclass(frozen=True)
class QChemBackend(BackendFormat):
    name: str = "qchem"

    def format_atom(self, atom: Atom, ghost: bool = False) -> str:
        # Common convention in Q-Chem: prefix element with '@' for ghost atoms.
        # If your environment uses a different syntax, override here.
        el = atom.element
        if ghost:
            el = "@" + el
        return f"{el:3s} {atom.x: .10f} {atom.y: .10f} {atom.z: .10f}"
