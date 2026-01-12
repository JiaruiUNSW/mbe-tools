from __future__ import annotations
import re
from .base import ParsedRecord

_RE_ENERGY = re.compile(r"^\s*FINAL\s+SINGLE\s+POINT\s+ENERGY\s+([-\d\.Ee\+]+)\s*$", re.MULTILINE)
_RE_CPU = re.compile(r"TOTAL\s+RUN\s+TIME:\s+(\d+)\s+days?\s+(\d+)\s+hours?\s+(\d+)\s+minutes?\s+([\d\.]+)\s+seconds", re.IGNORECASE)

def parse_orca_output(text: str, path: str, job_id: str = "") -> ParsedRecord:
    rec = ParsedRecord(
        job_id=job_id or path,
        program="orca",
        program_detected="orca",
        path=path,
        extra={},
    )

    m = _RE_ENERGY.search(text)
    if m:
        try:
            rec.energy_hartree = float(m.group(1))
            rec.status = rec.status or "ok"
        except Exception as exc:
            rec.error_reason = rec.error_reason or f"energy_parse_failed: {exc}"

    m = _RE_CPU.search(text)
    if m:
        try:
            d,h,mi,s = m.groups()
            rec.wall_seconds = int(d)*86400 + int(h)*3600 + int(mi)*60 + float(s)
            rec.cpu_seconds = rec.cpu_seconds or rec.wall_seconds
        except Exception as exc:
            rec.error_reason = rec.error_reason or f"runtime_parse_failed: {exc}"
    return rec
