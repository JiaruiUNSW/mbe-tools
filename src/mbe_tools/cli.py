from __future__ import annotations
import json
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING


if TYPE_CHECKING:
    from typer import Context


def _lazy_import_typer():
    try:
        import typer
        return typer
    except Exception as e:
        raise RuntimeError("CLI requires typer. Install with: pip install mbe-tools[cli]") from e


typer = _lazy_import_typer()
app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def _main(
    ctx: Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        is_eager=True,
        help="Show version and exit",
    ),
):
    """Global entrypoint for version flag."""
    if version:
        from . import __version__

        typer.echo(f"mbe-tools {__version__}")
        typer.echo(f"python: {sys.version.split()[0]}")
        typer.echo("jsonl schema: v1(calc-only), v2(cluster+calc)")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        return


@app.command()
def fragment(
    xyz_path: str = typer.Argument(..., help="Input XYZ file"),
    out_xyz: str = typer.Option("sample.xyz", help="Output XYZ file"),
    n: int = typer.Option(10, help="Number of fragments to sample"),
    seed: Optional[int] = typer.Option(None, help="Random seed"),
    require_ion: bool = typer.Option(False, help="Ensure at least one ion (non-water) is included if present"),
    mode: str = typer.Option("random", help="Sampling mode: random|spatial"),
    prefer_special: bool = typer.Option(False, help="For spatial mode, force inclusion of a special fragment if present"),
    k_neighbors: int = typer.Option(4, help="For spatial mode, candidate neighbor count when expanding"),
    start_index: Optional[int] = typer.Option(None, help="For spatial mode, starting fragment index"),
    oh_cutoff: float = typer.Option(1.25, help="O-H cutoff for water heuristic fragmentation (A)"),
):
    """Fragment a big cluster and sample N fragments."""
    from .cluster import read_xyz, fragment_by_water_heuristic, sample_fragments, spatial_sample_fragments, write_xyz

    xyz = read_xyz(xyz_path)
    frags = fragment_by_water_heuristic(xyz, oh_cutoff=oh_cutoff)
    mode_l = mode.lower()
    if mode_l == "spatial":
        picked = spatial_sample_fragments(
            frags,
            n=n,
            seed=seed,
            prefer_special=prefer_special or require_ion,
            k_neighbors=k_neighbors,
            start="index" if start_index is not None else ("special" if prefer_special or require_ion else "random"),
            start_index=start_index,
        )
    else:
        picked = sample_fragments(frags, n=n, seed=seed, require_ion=require_ion)
    write_xyz(out_xyz, picked, comment=f"sampled {n} fragments from {os.path.basename(xyz_path)}")
    typer.echo(f"Wrote: {out_xyz} (fragments={len(picked)})")


@app.command()
def gen(
    xyz_path: str = typer.Argument(..., help="Input XYZ file"),
    out_dir: str = typer.Option("mbe_geoms", help="Output directory"),
    max_order: int = typer.Option(2, help="Generate subsets up to this order"),
    orders: Optional[List[int]] = typer.Option(None, "--order", "--orders", help="Explicit subset orders (repeatable)"),
    cp: bool = typer.Option(True, help="Use CP-style ghosts for fragments not in subset"),
    scheme: str = typer.Option("mbe", help="MBE scheme/type label"),
    backend: str = typer.Option("qchem", help="Backend formatting: qchem/orca"),
    oh_cutoff: float = typer.Option(1.25, help="O-H cutoff for water heuristic fragmentation (A)"),
):
    """Generate subset geometries (coordinate blocks) for MBE jobs."""
    from .cluster import read_xyz, fragment_by_water_heuristic
    from .mbe import MBEParams, generate_subsets_xyz

    os.makedirs(out_dir, exist_ok=True)
    xyz = read_xyz(xyz_path)
    frags = fragment_by_water_heuristic(xyz, oh_cutoff=oh_cutoff)
    params = MBEParams(
        max_order=max_order,
        orders=orders,
        cp_correction=cp,
        backend=backend,
        scheme=scheme,
    )

    count = 0
    for job_id, subset, geom in generate_subsets_xyz(frags, params):
        k = len(subset)
        fn = os.path.join(out_dir, f"{job_id}_k{k}.geom")
        with open(fn, "w", encoding="utf-8") as f:
            f.write(geom + "\n")
        count += 1

    typer.echo(f"Generated {count} geometries into: {out_dir}")


@app.command()
def template(
    scheduler: str = typer.Option("pbs", help="pbs or slurm"),
    backend: str = typer.Option("qchem", help="qchem or orca"),
    job_name: str = typer.Option("mbe-job", help="Scheduler job name"),
    walltime: str = typer.Option("24:00:00", help="Walltime"),
    ncpus: int = typer.Option(16, help="PBS ncpus / Slurm cpus-per-task"),
    ntasks: int = typer.Option(1, help="Slurm ntasks (ignored for PBS)"),
    mem_gb: float = typer.Option(32.0, help="Memory in GB"),
    queue: Optional[str] = typer.Option(None, help="PBS queue"),
    project: Optional[str] = typer.Option(None, help="PBS project/account"),
    partition: Optional[str] = typer.Option(None, help="Slurm partition"),
    qos: Optional[str] = typer.Option(None, help="Slurm QoS"),
    chunk_size: Optional[int] = typer.Option(None, help="Inputs per child job (batch submit)"),
    module: Optional[str] = typer.Option(None, help="module load name"),
    command: Optional[str] = typer.Option(None, help="Executable command override"),
    out: str = typer.Option("job.sh", help="Output script"),
    wrapper: bool = typer.Option(False, help="Emit a bash submitter that writes hidden scheduler files then qsub/sbatch"),
):
    """Emit a simple PBS/Slurm script for Q-Chem or ORCA."""
    from .hpc_templates import render_pbs_qchem, render_slurm_orca

    sched = scheduler.lower()
    be = backend.lower()
    text: str
    if sched == "pbs" and be == "qchem":
        text = render_pbs_qchem(
            job_name=job_name,
            walltime=walltime,
            ncpus=ncpus,
            mem_gb=mem_gb,
            queue=queue,
            project=project,
            module=module or "qchem/5.2.2",
            chunk_size=chunk_size,
            wrapper=wrapper,
        )
    elif sched == "slurm" and be == "orca":
        text = render_slurm_orca(
            job_name=job_name,
            walltime=walltime,
            ntasks=ntasks,
            cpus_per_task=ncpus,
            mem_gb=mem_gb,
            partition=partition,
            account=project,
            qos=qos,
            module=module or "orca/5.0.3",
            command=command or "orca",
            chunk_size=chunk_size,
            wrapper=wrapper,
        )
    else:
        raise typer.BadParameter("Supported combinations: pbs+qchem, slurm+orca")

    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    typer.echo(f"Wrote template: {out}")


@app.command()
def build_input(
    geom: str = typer.Argument(..., help="Geometry block file (.geom or XYZ snippet)"),
    backend: str = typer.Option("qchem", help="qchem/orca"),
    method: str = typer.Option(..., help="Electronic structure method (e.g., wb97m-v)"),
    basis: str = typer.Option(..., help="Basis set (e.g., def2-ma-QZVPP)"),
    charge: int = typer.Option(0, help="Total charge"),
    multiplicity: int = typer.Option(1, help="Spin multiplicity"),
    thresh: Optional[float] = typer.Option(None, help="Q-Chem: THRESH"),
    tole: Optional[float] = typer.Option(None, help="Q-Chem: TolE"),
    scf_convergence: Optional[str] = typer.Option(None, help="SCF convergence keyword (qchem: scf_convergence; orca: TightSCF etc.)"),
    grid: Optional[str] = typer.Option(None, help="ORCA grid keyword (e.g., GRID5)"),
    rem_extra: Optional[str] = typer.Option(None, help="Extra Q-Chem $rem lines (newline-separated)"),
    keyword_line_extra: Optional[str] = typer.Option(None, help="Extra ORCA header keywords"),
    out: str = typer.Option("job.inp", help="Output input file"),
    glob_pattern: Optional[str] = typer.Option(None, "--glob", help="Batch mode: build inputs for all matching geom files in a directory"),
    out_dir: Optional[str] = typer.Option(None, help="Batch mode: output directory (defaults to geom directory)"),
):
    """Build a full input file from a geometry block."""
    from .input_builder import build_input_from_geom

    if glob_pattern:
        root = Path(geom)
        if not root.is_dir():
            raise typer.BadParameter("--glob requires that GEOM points to a directory")
        targets = sorted(root.glob(glob_pattern))
        if not targets:
            raise typer.BadParameter(f"No files match '{glob_pattern}' under {root}")
        out_base = Path(out_dir) if out_dir else root
        out_base.mkdir(parents=True, exist_ok=True)
        written = 0
        for geom_path in targets:
            text = build_input_from_geom(
                str(geom_path),
                backend=backend,
                method=method,
                basis=basis,
                charge=charge,
                multiplicity=multiplicity,
                thresh=thresh,
                tole=tole,
                scf_convergence=scf_convergence,
                grid=grid,
                rem_extra=rem_extra,
                keyword_line_extra=keyword_line_extra,
            )
            out_path = out_base / f"{geom_path.stem}.inp"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            written += 1
        typer.echo(f"Wrote {written} inputs to: {out_base}")
        return

    text = build_input_from_geom(
        geom,
        backend=backend,
        method=method,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        thresh=thresh,
        tole=tole,
        scf_convergence=scf_convergence,
        grid=grid,
        rem_extra=rem_extra,
        keyword_line_extra=keyword_line_extra,
    )
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    typer.echo(f"Wrote input: {out}")


# --- JSONL helpers ---


def _load_jsonl_with_cluster(path: str) -> Tuple[Optional[dict], List[dict]]:
    from .analysis import read_jsonl

    rows = read_jsonl(path)
    cluster = None
    calc_rows: List[dict] = []
    for r in rows:
        if isinstance(r, dict) and r.get("record_type") == "cluster":
            cluster = r
        else:
            calc_rows.append(r)
    return cluster, calc_rows


def _extract_combo(rec: dict) -> Tuple[Any, Any, Any, Any, Any]:
    return (
        rec.get("program"),
        rec.get("method"),
        rec.get("basis"),
        rec.get("grid"),
        rec.get("cp_correction"),
    )


def _combo_label(combo: Tuple[Any, Any, Any, Any, Any]) -> str:
    prog, method, basis, grid, cp = combo
    cp_s = "cp" if cp is True else ("nocp" if cp is False else "cp?")
    return f"{prog or '?'}|{method or '?'}|{basis or '?'}|{grid or '?'}|{cp_s}"


# --- New CLI commands (v0.2.0) ---


@app.command()
def show(
    jsonl_path: Optional[str] = typer.Argument(None, help="JSONL path (defaults apply)"),
    monomer: Optional[int] = typer.Option(None, help="Show details for monomer index"),
):
    """Quick human-readable view of JSONL (cluster + CPU + energy)."""
    from .jsonl_selector import select_jsonl

    path = select_jsonl(jsonl_path, echo=typer.echo)
    cluster, recs = _load_jsonl_with_cluster(path)

    typer.echo(f"JSONL: {path}")

    # Geometry
    if cluster:
        typer.echo(f"Cluster: id={cluster.get('cluster_id')} n_monomers={cluster.get('n_monomers')}")
        if monomer is not None:
            mons = cluster.get("monomers", [])
            if 0 <= monomer < len(mons):
                m = mons[monomer]
                typer.echo(f"Monomer {monomer} geometry:")
                for elem, x, y, z in m.get("geometry_xyz", []):
                    typer.echo(f"  {elem:2s} {x: .6f} {y: .6f} {z: .6f}")
            else:
                typer.echo(f"Monomer {monomer} not found in cluster record")
    else:
        typer.echo("Geometry: not available (no cluster record)")

    # CPU
    cpu_ok = sum(r.get("cpu_seconds", 0.0) or 0.0 for r in recs if r.get("status") in ("ok", None))
    cpu_all = sum(r.get("cpu_seconds", 0.0) or 0.0 for r in recs)
    ok_cnt = sum(1 for r in recs if r.get("status") in ("ok", None))
    fail_cnt = sum(1 for r in recs if r.get("status") not in ("ok", None))
    typer.echo(f"CPU: ok={cpu_ok:.2f}s all={cpu_all:.2f}s jobs ok/fail/total={ok_cnt}/{fail_cnt}/{len(recs)}")
    if monomer is not None:
        mono_recs = [r for r in recs if r.get("subset_size") == 1 and r.get("subset_indices") == [monomer]]
        if mono_recs:
            cpu_mono = sum(r.get("cpu_seconds", 0.0) or 0.0 for r in mono_recs)
            typer.echo(f"CPU (monomer {monomer}): {cpu_mono:.2f}s from {len(mono_recs)} jobs")

    # Energy summaries
    combos: Dict[Tuple[Any, Any, Any, Any, Any], int] = {}
    by_order: Dict[int, List[float]] = {}
    for r in recs:
        combo = _extract_combo(r)
        combos[combo] = combos.get(combo, 0) + 1
        k = r.get("subset_size")
        e = r.get("energy_hartree")
        if k is not None and e is not None:
            by_order.setdefault(int(k), []).append(float(e))
    typer.echo("Combinations:")
    for combo, cnt in combos.items():
        typer.echo(f"  {_combo_label(combo)}: {cnt} records")
    typer.echo("Energy by subset_size:")
    for k in sorted(by_order):
        vals = by_order[k]
        typer.echo(f"  k={k}: n={len(vals)} min={min(vals):.6f} max={max(vals):.6f} mean={sum(vals)/len(vals):.6f}")


@app.command()
def info(
    jsonl_path: Optional[str] = typer.Argument(None, help="JSONL path (defaults apply)"),
):
    """Summary panel of coverage, CPU, combinations, energies."""
    from .jsonl_selector import select_jsonl

    path = select_jsonl(jsonl_path, echo=typer.echo)
    cluster, recs = _load_jsonl_with_cluster(path)

    if cluster:
        typer.echo(f"Cluster: id={cluster.get('cluster_id')} n_monomers={cluster.get('n_monomers')} (geometry_incomplete={cluster.get('geometry_incomplete')})")
    else:
        typer.echo("Cluster: not available (no cluster record)")

    combos: Dict[Tuple[Any, Any, Any, Any, Any], int] = {}
    by_order_status: Dict[int, Dict[str, int]] = {}
    cpu_ok = 0.0
    cpu_all = 0.0
    for r in recs:
        combo = _extract_combo(r)
        combos[combo] = combos.get(combo, 0) + 1
        k = r.get("subset_size")
        st = r.get("status") or "ok"
        by_order_status.setdefault(int(k) if k is not None else -1, {}).setdefault(st, 0)
        by_order_status[int(k) if k is not None else -1][st] += 1
        cpu_all += (r.get("cpu_seconds") or 0.0)
        if st == "ok":
            cpu_ok += (r.get("cpu_seconds") or 0.0)

    typer.echo("Combinations:")
    for combo, cnt in combos.items():
        typer.echo(f"  {_combo_label(combo)}: {cnt} records")

    typer.echo("Coverage by subset_size (status counts):")
    for k in sorted(by_order_status):
        st_map = by_order_status[k]
        parts = ", ".join(f"{s}:{n}" for s, n in st_map.items())
        typer.echo(f"  k={k}: {parts}")

    typer.echo(f"CPU: ok={cpu_ok:.2f}s all={cpu_all:.2f}s")


@app.command()
def calc(
    jsonl_path: Optional[str] = typer.Argument(None, help="JSONL path (defaults apply)"),
    scheme: str = typer.Option("simple", help="simple|strict"),
    to: Optional[int] = typer.Option(None, help="Compute up to order K"),
    from_order: Optional[int] = typer.Option(None, "--from", help="Lower order for ΔE(i→j)"),
    monomer: Optional[int] = typer.Option(None, help="Report monomer energy (subset_size=1, index)"),
    unit: str = typer.Option("hartree", help="hartree|kcal|kj"),
):
    """Compute CPU totals and MBE energies."""
    from .jsonl_selector import select_jsonl
    from .analysis import strict_mbe_orders

    path = select_jsonl(jsonl_path, echo=typer.echo)
    cluster, recs = _load_jsonl_with_cluster(path)

    # Validate unit
    unit_l = unit.lower()
    unit_factor = {"hartree": 1.0, "kcal": 627.509474, "kj": 2625.49962}.get(unit_l)
    if unit_factor is None:
        raise typer.BadParameter("--unit must be hartree|kcal|kj")

    # Prevent mixed program/method/basis/grid/cp combos (ignore None values)
    combo_seen: list[Any] = [None, None, None, None, None]
    mixed_combo = False
    for r in recs:
        combo = _extract_combo(r)
        for i, val in enumerate(combo):
            if val is None:
                continue
            if combo_seen[i] is None:
                combo_seen[i] = val
            elif combo_seen[i] != val:
                mixed_combo = True
                break
        if mixed_combo:
            break
    if mixed_combo:
        typer.echo("Mixed program/method/basis/grid/cp combinations detected; please split files")
        raise typer.Exit(code=1)

    # CPU
    cpu_ok = sum(r.get("cpu_seconds", 0.0) or 0.0 for r in recs if r.get("status") in ("ok", None))
    cpu_all = sum(r.get("cpu_seconds", 0.0) or 0.0 for r in recs)
    typer.echo(f"CPU: ok={cpu_ok:.2f}s all={cpu_all:.2f}s")

    # Monomer energy
    if monomer is not None:
        mono = [r for r in recs if r.get("subset_size") == 1 and r.get("subset_indices") == [monomer] and r.get("energy_hartree") is not None]
        if mono:
            e = mono[0]["energy_hartree"]
            typer.echo(f"E(monomer {monomer}) = {e * unit_factor:.10f} {unit_l}")
        else:
            typer.echo(f"Monomer {monomer} energy not found")

    # Energy aggregation
    if scheme.lower() == "strict":
        rows, missing = strict_mbe_orders(recs, max_order=to)
        if rows:
            typer.echo("MBE (strict inclusion–exclusion):")
            for r in rows:
                typer.echo(f"  order={r['order']} E={r['energy_hartree'] * unit_factor:.10f} {unit_l}")
        if missing:
            typer.echo(f"Missing lower-order subsets: {sorted(set(missing))}")
    else:
        # simple: mean by order and optional ΔE(from→to)
        by_order: Dict[int, List[float]] = {}
        for r in recs:
            k = r.get("subset_size")
            e = r.get("energy_hartree")
            if k is None or e is None:
                continue
            by_order.setdefault(int(k), []).append(float(e))
        if by_order:
            ref = sum(by_order.get(1, [])) / len(by_order.get(1, [])) if by_order.get(1) else 0.0
            typer.echo("Mean energies:")
            for k in sorted(by_order):
                vals = by_order[k]
                mean_e = sum(vals) / len(vals)
                typer.echo(f"  k={k}: mean={mean_e * unit_factor:.10f} {unit_l}; ΔE vs mean(k=1)={(mean_e - ref) * unit_factor:.10f} {unit_l}")
            if to is not None and from_order is not None:
                if to in by_order and from_order in by_order:
                    delta = (sum(by_order[to]) / len(by_order[to])) - (sum(by_order[from_order]) / len(by_order[from_order]))
                    typer.echo(f"ΔE(k={from_order}→{to}) = {delta * unit_factor:.10f} {unit_l}")
                else:
                    typer.echo("ΔE request skipped (orders missing)")


@app.command()
def save(
    jsonl_path: Optional[str] = typer.Argument(None, help="JSONL path (defaults apply)"),
    dest: str = typer.Option(..., help="Destination directory"),
):
    """Archive JSONL (copy) with timestamped folder."""
    from .jsonl_selector import select_jsonl

    path = select_jsonl(jsonl_path, echo=typer.echo)
    dest_dir = Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    cluster, _ = _load_jsonl_with_cluster(path)
    cluster_id = (cluster.get("cluster_id") if cluster else None) or "unknown"
    out_dir = dest_dir / cluster_id / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / Path(path).name
    shutil.copy2(path, target)
    typer.echo(f"Saved to {target}")


@app.command()
def compare(
    path_or_glob: str = typer.Argument(..., help="Directory or glob for *.jsonl"),
    cluster: Optional[str] = typer.Option(None, help="Filter cluster_id"),
):
    """Compare multiple JSONL runs (CPU + counts)."""
    import glob as _glob

    paths: List[str] = []
    p = Path(path_or_glob)
    if p.is_dir():
        paths = [str(x) for x in p.rglob("*.jsonl")]
    else:
        paths = _glob.glob(path_or_glob, recursive=True)
    if not paths:
        raise typer.BadParameter("No JSONL files found for compare")

    rows = []
    for path in sorted(paths):
        cl, recs = _load_jsonl_with_cluster(path)
        cid = cl.get("cluster_id") if cl else "unknown"
        if cluster and cid != cluster:
            continue
        cpu_ok = sum(r.get("cpu_seconds", 0.0) or 0.0 for r in recs if r.get("status") in ("ok", None))
        counts = len(recs)
        combos = {_combo_label(_extract_combo(r)) for r in recs}
        rows.append((cid, Path(path).name, cpu_ok, counts, ",".join(sorted(combos))[:80]))

    if not rows:
        raise typer.BadParameter("No matching JSONL after filters")

    typer.echo("cluster_id | file | cpu_ok(s) | records | combos")
    for cid, fname, cpu_ok, cnt, combos in rows:
        typer.echo(f"{cid} | {fname} | {cpu_ok:.2f} | {cnt} | {combos}")


@app.command()
def parse(
    root: str = typer.Argument(..., help="Root directory containing outputs"),
    program: str = typer.Option("qchem", help="qchem/orca/auto"),
    glob_pattern: str = typer.Option("*.out", "--glob-pattern", "--glob", help="Glob pattern, e.g. '*.out'"),
    out: str = typer.Option("parsed.jsonl", help="Output JSONL file"),
    infer_metadata: bool = typer.Option(True, help="Infer subset/method/basis metadata from paths"),
    cluster_xyz: Optional[str] = typer.Option(None, help="Provide supersystem XYZ to embed as cluster record"),
    nosearch: bool = typer.Option(False, help="Do not search .out for geometry; emit calc-only JSONL"),
    geom_max_lines: int = typer.Option(5000, help="Scan first N lines for geometry blocks"),
    geom_mode: str = typer.Option("first", help="Geometry block pick: first|last"),
    geom_source: str = typer.Option("singleton", help="Geometry extraction source: singleton|any"),
    geom_drop_ghost: bool = typer.Option(True, help="Drop ghost atoms when extracting geometry"),
):
    """Parse output files to JSONL."""
    from .parsers.io import glob_paths, parse_files
    from .cluster import read_xyz, fragment_by_water_heuristic
    from .utils import Atom

    paths = glob_paths(root, glob_pattern)
    recs = parse_files(paths, program=program, infer_metadata=infer_metadata)

    def _read_head(path: str, n: int) -> list[str]:
        lines: list[str] = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for i, ln in enumerate(f):
                    if i >= n:
                        break
                    lines.append(ln.rstrip("\n"))
        except OSError:
            return []
        return lines

    def _is_ghost(elem: str) -> bool:
        up = elem.upper()
        return up in {"BQ", "GH", "X", "XX", "Q"} or up.startswith("GH") or up.startswith("BQ")

    def _parse_block(lines: list[str], start: int) -> tuple[list[Atom], int]:
        atoms: list[Atom] = []
        i = start
        while i < len(lines):
            parts = lines[i].split()
            if len(parts) < 4:
                break
            # handle optional leading index
            if parts[0].isdigit():
                parts = parts[1:]
            if len(parts) < 4:
                break
            el = parts[0]
            try:
                x, y, z = map(float, parts[1:4])
            except Exception:
                break
            if geom_drop_ghost and _is_ghost(el):
                i += 1
                continue
            atoms.append(Atom(el, x, y, z))
            i += 1
        return atoms, i

    def extract_geometry_from_out_head(path: str, prog: str) -> Optional[list[Atom]]:
        lines = _read_head(path, geom_max_lines)
        if not lines:
            return None
        matches: list[tuple[int, list[Atom]]] = []
        prog_l = prog.lower()
        for idx, ln in enumerate(lines):
            ln_l = ln.lower()
            if prog_l in ("qchem", "q-chem"):
                if "standard nuclear orientation" in ln_l or "coordinates (angstroms)" in ln_l:
                    # skip header/separator lines following
                    j = idx + 1
                    # skip until dashed line
                    while j < len(lines) and (not lines[j].strip() or set(lines[j].strip()) <= {"-", "="}):
                        j += 1
                    atoms, end_idx = _parse_block(lines, j)
                    if atoms:
                        matches.append((idx, atoms))
            elif prog_l == "orca":
                if "cartesian coordinates (angstrom" in ln_l:
                    j = idx + 1
                    # skip header lines until blank
                    while j < len(lines) and lines[j].strip():
                        j += 1
                    # skip possible blank
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    atoms, end_idx = _parse_block(lines, j)
                    if atoms:
                        matches.append((idx, atoms))
        if not matches:
            return None
        if geom_mode.lower() == "last":
            return matches[-1][1]
        return matches[0][1]

    cluster_record: Optional[dict] = None

    if cluster_xyz:
        xyz = read_xyz(cluster_xyz)
        frags = fragment_by_water_heuristic(xyz)
        monomers = []
        for i, frag in enumerate(frags):
            monomers.append(
                {
                    "monomer_index": i,
                    "label": frag.label,
                    "charge": frag.charge_hint,
                    "multiplicity": None,
                    "geometry_xyz": [[a.element, a.x, a.y, a.z] for a in frag.atoms],
                }
            )
        cluster_record = {
            "record_type": "cluster",
            "schema_version": 2,
            "cluster_id": Path(cluster_xyz).stem,
            "source": {"type": "cluster_xyz", "path": str(cluster_xyz)},
            "unit": "angstrom",
            "n_monomers": len(monomers),
            "monomers": monomers,
            "geometry_incomplete": False,
            "missing_monomers": [],
            "extra": {},
        }
    elif not nosearch:
        geom_source_mode = geom_source.lower()
        if geom_source_mode not in {"singleton", "any"}:
            raise typer.BadParameter("--geom-source must be singleton or any")

        def _monomer_records() -> list:
            eligible: list = []
            for r in recs:
                if r.status not in ("ok", None):
                    continue
                if not r.subset_indices or len(r.subset_indices) != 1:
                    continue
                if geom_source_mode == "singleton" and r.subset_size not in (None, 1):
                    continue
                eligible.append(r)
            return eligible

        singleton_recs = _monomer_records()
        geom_by_idx: dict[int, list[Atom]] = {}
        for r in singleton_recs:
            m_idx = r.subset_indices[0]
            atoms = extract_geometry_from_out_head(r.path, r.program or program)
            if atoms:
                geom_by_idx.setdefault(m_idx, atoms)

        # Fallback: if no singleton indices are available, try the first parsable geometry as monomer 0
        if not geom_by_idx:
            for r in recs:
                if r.status not in ("ok", None):
                    continue
                atoms = extract_geometry_from_out_head(r.path, r.program or program)
                if atoms:
                    geom_by_idx[0] = atoms
                    break
        if geom_by_idx:
            max_idx = max(geom_by_idx.keys())
            n_monomers = max_idx + 1
            missing = [i for i in range(n_monomers) if i not in geom_by_idx]
            monomers = []
            for i in range(n_monomers):
                atoms = geom_by_idx.get(i, [])
                monomers.append(
                    {
                        "monomer_index": i,
                        "label": None,
                        "charge": None,
                        "multiplicity": None,
                        "geometry_xyz": [[a.element, a.x, a.y, a.z] for a in atoms] if atoms else [],
                    }
                )
            cluster_record = {
                "record_type": "cluster",
                "schema_version": 2,
                "cluster_id": Path(root).name,
                "source": {"type": "out_search", "path": str(root)},
                "unit": "angstrom",
                "n_monomers": n_monomers,
                "monomers": monomers,
                "geometry_incomplete": bool(missing),
                "missing_monomers": missing,
                "extra": {},
            }

    with open(out, "w", encoding="utf-8") as f:
        if cluster_record:
            f.write(json.dumps(cluster_record, ensure_ascii=False) + "\n")
        for r in recs:
            f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")

    typer.echo(f"Parsed {len(recs)} files → {out}")
    if cluster_record:
        missing = cluster_record.get("missing_monomers", [])
        if missing:
            typer.echo(f"Geometry: embedded but incomplete (missing={missing})")
        else:
            typer.echo("Geometry: embedded (cluster record written)")
    elif nosearch:
        typer.echo("Geometry: skipped (--nosearch)")
    else:
        typer.echo("Geometry: not found (no cluster record written)")


@app.command()
def analyze(
    jsonl_path: Optional[str] = typer.Argument(None, help="Input JSONL (defaults: run.jsonl → parsed.jsonl → single *.jsonl → newest)"),
    to_csv: Optional[str] = typer.Option(None, help="Write full table to CSV"),
    to_xlsx: Optional[str] = typer.Option(None, help="Write full table to Excel"),
    plot: Optional[str] = typer.Option(None, help="Plot delta energy (requires matplotlib)"),
    scheme: str = typer.Option("simple", help="Energy aggregation: simple|strict (inclusion–exclusion)"),
    max_order: Optional[int] = typer.Option(None, help="Maximum order for strict aggregation"),
):
    """Analyze parsed JSONL (basic summaries + exports)."""
    from .analysis import read_jsonl, to_dataframe, summarize_by_order, compute_delta_energy, strict_mbe_orders
    from .jsonl_selector import select_jsonl

    path = select_jsonl(jsonl_path, echo=typer.echo)

    records = read_jsonl(path)
    df = to_dataframe(records)

    scheme_l = scheme.lower()
    if scheme_l == "strict":
        mbe_rows, missing = strict_mbe_orders(records, max_order=max_order)
        if mbe_rows:
            import pandas as pd
            mbe_df = pd.DataFrame(mbe_rows)
            typer.echo("Inclusion–exclusion MBE(k):")
            typer.echo(mbe_df.to_string(index=False))
        if missing:
            typer.echo(f"Warning: missing lower-order subsets encountered: {sorted(set(missing))}")
    else:
        df = compute_delta_energy(df)
        summ = summarize_by_order(df)
        typer.echo(summ.to_string(index=False))

    if to_csv:
        df.to_csv(to_csv, index=False)
        typer.echo(f"Wrote CSV: {to_csv}")

    if to_xlsx:
        df.to_excel(to_xlsx, index=False)
        typer.echo(f"Wrote Excel: {to_xlsx}")

    if plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise RuntimeError("Plot requires matplotlib. Install with: pip install mbe-tools[analysis]") from e
        plt.figure()
        plt.scatter(df["subset_size"], df.get("delta_energy_hartree_vs_ref", df["energy_hartree"]))
        plt.xlabel("subset_size")
        ylabel = "ΔE (Hartree) vs mean(order=1)" if scheme_l == "simple" else "MBE energy (Hartree)"
        plt.ylabel(ylabel)
        plt.savefig(plot, dpi=200, bbox_inches="tight")
        typer.echo(f"Wrote plot: {plot}")
