from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class ParsedRecord:
    job_id: str
    program: str
    path: str
    program_detected: Optional[str] = None
    status: Optional[str] = None
    error_reason: Optional[str] = None
    energy_hartree: Optional[float] = None
    cpu_seconds: Optional[float] = None
    wall_seconds: Optional[float] = None
    method: Optional[str] = None
    basis: Optional[str] = None
    grid: Optional[str] = None

    subset_size: Optional[int] = None
    subset_indices: Optional[List[int]] = None
    cp_correction: Optional[bool] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "program": self.program,
            "program_detected": self.program_detected,
            "status": self.status,
            "error_reason": self.error_reason,
            "path": self.path,
            "energy_hartree": self.energy_hartree,
            "cpu_seconds": self.cpu_seconds,
            "wall_seconds": self.wall_seconds,
            "method": self.method,
            "basis": self.basis,
            "grid": self.grid,
            "subset_size": self.subset_size,
            "subset_indices": self.subset_indices,
            "cp_correction": self.cp_correction,
            "extra": self.extra or {},
        }
