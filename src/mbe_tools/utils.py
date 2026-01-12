from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Dict, Any
import math
import itertools
import random

@dataclass(frozen=True)
class Atom:
    element: str
    x: float
    y: float
    z: float

def distance(a: Atom, b: Atom) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def chunked(it: Iterable[Any], n: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

def powerset_indices(indices: List[int], k: int):
    return itertools.combinations(indices, k)
