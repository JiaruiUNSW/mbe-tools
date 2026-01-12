from __future__ import annotations
from dataclasses import dataclass
from ..utils import Atom
from .base import BackendFormat

@dataclass(frozen=True)
class OrcaBackend(BackendFormat):
    name: str = "orca"

    def format_atom(self, atom: Atom, ghost: bool = False) -> str:
        # ORCA ghost-atom syntax varies by input style.
        # A commonly used pattern is to add ':' after the element, e.g., 'H : x y z'
        # If your workflow differs, override this method.
        el = atom.element
        if ghost:
            el = el + " :"
            return f"{el:3s} {atom.x: .10f} {atom.y: .10f} {atom.z: .10f}"
        return f"{el:2s} {atom.x: .10f} {atom.y: .10f} {atom.z: .10f}"
