from __future__ import annotations
import re
from typing import Optional
from .base import ParsedRecord

_RE_ENERGY = re.compile(r"^\s*Total\s+energy\s+in\s+the\s+final\s+basis\s+set\s*=\s*([-\d\.Ee\+]+)", re.MULTILINE)
_RE_ENERGY2 = re.compile(r"^\s*SCF\s+energy\s*=\s*([-\d\.Ee\+]+)", re.MULTILINE)
_RE_CPU = re.compile(r"^\s*Total\s+job\s+time:\s*([\d\.]+)\s*s", re.MULTILINE)

def parse_qchem_output(text: str, path: str, job_id: str = "") -> ParsedRecord:
    rec = ParsedRecord(
        job_id=job_id or path,
        program="qchem",
        program_detected="qchem",
        path=path,
        extra={},
    )

    m = _RE_ENERGY.search(text) or _RE_ENERGY2.search(text)
    if m:
        try:
            rec.energy_hartree = float(m.group(1))
            rec.status = rec.status or "ok"
        except Exception as exc:
            rec.error_reason = rec.error_reason or f"energy_parse_failed: {exc}"

    m = _RE_CPU.search(text)
    if m:
        try:
            rec.cpu_seconds = float(m.group(1))
            rec.wall_seconds = rec.wall_seconds or rec.cpu_seconds
        except Exception as exc:
            rec.error_reason = rec.error_reason or f"cpu_time_parse_failed: {exc}"

    # method/basis/grid are often in the input, not always echoed in output; keep as optional
    return rec
