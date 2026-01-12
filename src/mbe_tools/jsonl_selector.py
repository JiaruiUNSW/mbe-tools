from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable


def _safe_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def select_jsonl(
    jsonl_path: Optional[str],
    *,
    cwd: Optional[str] = None,
    echo: Optional[Callable[[str], None]] = None,
) -> str:
    """Select a JSONL path following default priority rules.

    Priority:
    1) explicit path if provided (validated)
    2) ./run.jsonl
    3) ./parsed.jsonl
    4) single *.jsonl in cwd
    5) newest *.jsonl if multiple (announce and list candidates up to 10)
    """
    base = Path(cwd or ".").resolve()

    def _emit(msg: str) -> None:
        if echo:
            echo(msg)

    if jsonl_path:
        p = Path(jsonl_path)
        if not p.is_absolute():
            p = base / p
        if not p.is_file():
            raise FileNotFoundError(f"JSONL not found: {p}")
        return str(p)

    preferred = [base / "run.jsonl", base / "parsed.jsonl"]
    for p in preferred:
        if p.is_file():
            return str(p)

    jsonls = [p for p in base.glob("*.jsonl") if p.is_file()]
    if not jsonls:
        raise FileNotFoundError("No JSONL found. Please provide a path.")
    if len(jsonls) == 1:
        return str(jsonls[0])

    # Multiple: pick newest, list top candidates
    jsonls_sorted = sorted(jsonls, key=_safe_mtime, reverse=True)
    newest = jsonls_sorted[0]
    lines = [f"Auto-selected: {newest} (newest)"]
    top = jsonls_sorted[:10]
    if len(top) > 1:
        lines.append("Candidates:")
        for p in top:
            lines.append(f"  {p.name}")
    _emit("\n".join(lines))
    return str(newest)
