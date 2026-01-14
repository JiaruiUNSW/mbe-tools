from __future__ import annotations
from typing import Optional


def _read_geom(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def render_qchem_input(
    geom_block: str,
    *,
    method: str,
    basis: str,
    charge: int = 0,
    multiplicity: int = 1,
    thresh: Optional[float] = None,
    tole: Optional[float] = None,
    scf_convergence: Optional[str] = None,
    xc_grid: Optional[str] = None,
    rem_extra: Optional[str] = None,
) -> str:
    """Render a minimal Q-Chem input from a geometry block."""
    lines = [
        "$molecule",
        f"{charge} {multiplicity}",
        geom_block.strip(),
        "$end",
        "",
        "$rem",
        f"  method        {method}",
        f"  basis         {basis}",
    ]
    if thresh is not None:
        lines.append(f"  thresh        {thresh:g}")
    if tole is not None:
        lines.append(f"  tole          {tole:g}")
    if scf_convergence is not None:
        lines.append(f"  scf_convergence {scf_convergence}")
    if xc_grid is not None:
        lines.append(f"  xc_grid       {xc_grid}")
    if rem_extra:
        for ln in rem_extra.strip().splitlines():
            ln = ln.strip()
            if ln:
                lines.append(f"  {ln}")
    lines.append("$end")
    return "\n".join(lines) + "\n"


def render_orca_input(
    geom_block: str,
    *,
    method: str,
    basis: str,
    charge: int = 0,
    multiplicity: int = 1,
    grid: Optional[str] = None,
    scf_convergence: Optional[str] = None,
    keyword_line_extra: Optional[str] = None,
) -> str:
    """Render a minimal ORCA input from a geometry block."""
    header_parts = [method, basis]
    if grid:
        header_parts.append(grid)
    if scf_convergence:
        header_parts.append(scf_convergence)
    if keyword_line_extra:
        header_parts.append(keyword_line_extra.strip())
    header = "! " + " ".join(header_parts)
    lines = [
        header,
        f"* xyz {charge} {multiplicity}",
        geom_block.strip(),
        "*",
    ]
    return "\n".join(lines) + "\n"


def build_input_from_geom(
    geom_path: str,
    *,
    backend: str,
    method: str,
    basis: str,
    charge: int = 0,
    multiplicity: int = 1,
    thresh: Optional[float] = None,
    tole: Optional[float] = None,
    scf_convergence: Optional[str] = None,
    xc_grid: Optional[str] = None,
    grid: Optional[str] = None,
    rem_extra: Optional[str] = None,
    keyword_line_extra: Optional[str] = None,
) -> str:
    geom = _read_geom(geom_path)
    name = backend.lower()
    if name in ("qchem", "q-chem"):
        return render_qchem_input(
            geom,
            method=method,
            basis=basis,
            charge=charge,
            multiplicity=multiplicity,
            thresh=thresh,
            tole=tole,
            scf_convergence=scf_convergence,
            xc_grid=xc_grid,
            rem_extra=rem_extra,
        )
    if name == "orca":
        return render_orca_input(
            geom,
            method=method,
            basis=basis,
            charge=charge,
            multiplicity=multiplicity,
            grid=grid,
            scf_convergence=scf_convergence,
            keyword_line_extra=keyword_line_extra,
        )
    raise ValueError(f"Unknown backend for input build: {backend}")
