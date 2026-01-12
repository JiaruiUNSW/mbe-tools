from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Any
import glob
import os
import re

from .qchem import parse_qchem_output
from .orca import parse_orca_output
from .base import ParsedRecord


_METHOD_PATTERNS = [
    r"wb97m?-v",
    r"wb97x?-v",
    r"pbe0?",
    r"b3lyp",
    r"m06-2x",
    r"mp2",
    r"ccsd\(t\)",
]

_BASIS_PATTERNS = [
    r"def2-[a-z0-9\-\+]+",
    r"aug-cc-pv[dtq56]z[p]?",
    r"cc-pv[dtq56]z[p]?",
    r"6-31g\+?\+?\([dp]\)",
    r"6-311g\+?\+?\([dp]\)",
]


def detect_program(text: str) -> str:
    """Best-effort program detection from output text."""
    if re.search(r"Q-Chem|QCHEM|Q\s*CHEM", text, flags=re.IGNORECASE):
        return "qchem"
    if re.search(r"\bORCA\b|O\s+R\s+C\s+A", text, flags=re.IGNORECASE):
        return "orca"
    return "unknown"


def infer_metadata_from_path(path: str) -> Dict[str, Any]:
    """Best-effort metadata inference from filenames/directories.

    Heuristics (all optional):
    - subset_size: look for tokens like 'k2', 'order3'
    - subset_indices: tokens like 'f0-3-9' / 'idx0_3_9'
    - cp_correction: 'cp' / 'nocp'
    - method/basis/grid: match common tokens if present
    """
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    lower = stem.lower()
    meta: Dict[str, Any] = {}

    m = re.search(r"k(\d+)", lower)
    if m:
        meta["subset_size"] = int(m.group(1))

    # Preferred pattern: ..._k2_f000-003-007_cp.out (indices already 0-based, allow zero padding)
    m = re.search(r"(?:f|idx|subset)[-_]?((?:\d+[-_])+\d+)", lower)
    if m:
        nums = re.findall(r"\d+", m.group(1))
        meta["subset_indices"] = [int(x) for x in nums]

    # Legacy pattern: ..._k2_1.3_abcd1234 (1-based indices separated by '.')
    m = re.search(r"k\d+_((?:\d+\.)+\d+)", lower)
    if m and "subset_indices" not in meta:
        nums = re.findall(r"\d+", m.group(1))
        meta["subset_indices"] = [int(x) - 1 for x in nums]  # convert to 0-based

    if "nocp" in lower or "no_cp" in lower or "no-cp" in lower:
        meta["cp_correction"] = False
    elif "cp" in lower:
        meta["cp_correction"] = True

    for pat in _METHOD_PATTERNS:
        m = re.search(pat, stem, flags=re.IGNORECASE)
        if m:
            meta["method"] = m.group(0)
            break

    for pat in _BASIS_PATTERNS:
        m = re.search(pat, stem, flags=re.IGNORECASE)
        if m:
            meta["basis"] = m.group(0)
            break

    return meta


def _apply_inferred_metadata(rec: ParsedRecord, meta: Dict[str, Any]) -> None:
    if rec.subset_size is None and meta.get("subset_size") is not None:
        rec.subset_size = meta["subset_size"]
    if rec.subset_indices is None and meta.get("subset_indices") is not None:
        rec.subset_indices = meta["subset_indices"]
    if rec.cp_correction is None and meta.get("cp_correction") is not None:
        rec.cp_correction = meta["cp_correction"]
    if rec.method is None and meta.get("method") is not None:
        rec.method = meta["method"]
    if rec.basis is None and meta.get("basis") is not None:
        rec.basis = meta["basis"]


def _read_text(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return None


def _parse_qchem_input_metadata(text: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    m = re.search(r"^\s*method\s+([\w\-]+)", text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        meta["method"] = m.group(1)
    m = re.search(r"^\s*basis\s+([\w\-\+]+)", text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        meta["basis"] = m.group(1)
    m = re.search(r"^\s*xc_functional\s+([\w\-]+)", text, flags=re.IGNORECASE | re.MULTILINE)
    if m and "method" not in meta:
        meta["method"] = m.group(1)
    return meta


def _parse_orca_input_metadata(text: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    header_match = re.search(r"^!\s+(.+)$", text, flags=re.MULTILINE)
    if header_match:
        tokens = header_match.group(1).split()
        if tokens:
            meta["method"] = tokens[0]
        if len(tokens) > 1:
            meta["basis"] = tokens[1]
        # look for grid tokens
        for tok in tokens[2:]:
            if tok.upper().startswith("GRID"):
                meta["grid"] = tok
                break
    return meta


def _infer_from_companion_input(rec: ParsedRecord, program: str) -> None:
    stem, _ = os.path.splitext(rec.path)
    candidates = [stem + ".inp", stem + ".in"]
    for c in candidates:
        text = _read_text(c)
        if not text:
            continue
        meta = _parse_qchem_input_metadata(text) if program in ("qchem", "q-chem") else _parse_orca_input_metadata(text)
        _apply_inferred_metadata(rec, meta)
        break

def parse_files(paths: Iterable[str], program: str, infer_metadata: bool = True) -> List[ParsedRecord]:
    out: List[ParsedRecord] = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        base_prog = program.lower()
        detected = detect_program(text) if base_prog == "auto" else base_prog
        final_prog = detected if base_prog == "auto" else base_prog
        if final_prog == "unknown":
            raise ValueError(f"Program auto-detect failed for: {p}")

        if final_prog in ("qchem", "q-chem"):
            rec = parse_qchem_output(text, p, job_id=os.path.basename(p))
        elif final_prog in ("orca",):
            rec = parse_orca_output(text, p, job_id=os.path.basename(p))
        else:
            raise ValueError(f"Unknown program: {program}")

        rec.program = "qchem" if final_prog in ("qchem", "q-chem") else "orca"
        rec.program_detected = detected if detected != "unknown" else rec.program_detected
        if infer_metadata:
            meta = infer_metadata_from_path(p)
            _apply_inferred_metadata(rec, meta)
            if (rec.method is None or rec.basis is None or rec.grid is None) and os.path.isfile(p):
                _infer_from_companion_input(rec, final_prog)
        out.append(rec)
    return out

def glob_paths(root: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, pattern), recursive=True))
