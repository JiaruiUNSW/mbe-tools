from __future__ import annotations
import csv
import json
import math
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING

from .utils import Atom


def _lazy_import_typer():
    try:
        import typer
        return typer
    except Exception as e:
        raise RuntimeError("CLI requires typer. Install with: pip install mbe-tools[cli]") from e


if TYPE_CHECKING:
    import typer  # type: ignore
else:
    typer = _lazy_import_typer()
TyperContext = typer.Context
app = typer.Typer(add_completion=False)


def _xdg_paths() -> Dict[str, Path]:
    home = Path.home()
    return {
        "data": home / ".local" / "share" / "mbe-tools",
        "config": home / ".config" / "mbe-tools",
        "cache": home / ".cache" / "mbe-tools",
        "state": home / ".local" / "state" / "mbe-tools",
    }


def _library_config_path() -> Path:
    paths = _xdg_paths()
    paths["config"].mkdir(parents=True, exist_ok=True)
    return paths["config"] / "library_path.txt"
def _resolve_library_root(dest: Optional[str]) -> Path:
    """Resolve the base directory for saved runs.

    Priority: explicit --dest > env MBE_SAVE_DEST > config file > XDG data/runs
    """
    xdg = _xdg_paths()
    default_runs = xdg["data"] / "runs"

    if dest:
        return Path(dest).expanduser()

    env_dest = os.getenv("MBE_SAVE_DEST")
    if env_dest:
        return Path(env_dest).expanduser()

    cfg = _library_config_path()
    if cfg.is_file():
        try:
            text = cfg.read_text(encoding="utf-8").strip()
            if text:
                return Path(text).expanduser()
        except OSError:
            # If the library config file cannot be read (permissions, corruption, etc.),
            # ignore the error and fall back to the default runs directory.
            pass

    return default_runs


def _load_fragments_from_monomer_dir(root: Path, pattern: str) -> List[List[Atom]]:
    paths = sorted(root.glob(pattern))
    if not paths:
        raise typer.BadParameter(f"No monomer geoms match '{pattern}' under {root}")

    def _read_fragment_from_geom(path: Path) -> List[Atom]:
        atoms: List[Atom] = []
        with path.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                if len(parts) < 4:
                    continue
                el = parts[0].lstrip("@")
                if not el:
                    continue
                try:
                    x, y, z = map(float, parts[1:4])
                except ValueError as e:
                    raise typer.BadParameter(f"Invalid coord line in {path}: {ln}") from e
                atoms.append(Atom(el, x, y, z))
        if not atoms:
            raise typer.BadParameter(f"Monomer geom {path} yielded no atoms")
        return atoms

    return [_read_fragment_from_geom(p) for p in paths]


@app.callback(invoke_without_command=True)
def _main(
    ctx: TyperContext,
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
    monomers_dir: Optional[str] = typer.Option(None, help="Directory of monomer .geom files to assemble higher-order subsets"),
    monomer_glob: str = typer.Option("*.geom", help="Glob to select monomer .geom files (used with --monomers-dir)"),
    cluster_name: Optional[str] = typer.Option(None, help="Use as filename prefix (fallback to backend if omitted)"),
):
    """Generate subset geometries (coordinate blocks) for MBE jobs."""
    from .cluster import read_xyz, fragment_by_water_heuristic
    from .mbe import MBEParams, generate_subsets_xyz

    os.makedirs(out_dir, exist_ok=True)
    if monomers_dir:
        root = Path(monomers_dir)
        if not root.is_dir():
            raise typer.BadParameter(f"--monomers-dir must be a directory: {monomers_dir}")
        frags = _load_fragments_from_monomer_dir(root, monomer_glob)
    else:
        xyz = read_xyz(xyz_path)
        frags = fragment_by_water_heuristic(xyz, oh_cutoff=oh_cutoff)
    params = MBEParams(
        max_order=max_order,
        orders=orders,
        cp_correction=cp,
        backend=backend,
        name_prefix=cluster_name or None,
        scheme=scheme,
    )

    count = 0
    for job_id, subset, geom in generate_subsets_xyz(frags, params):
        k = len(subset)
        # File name uses backend + order + 1-based indices separated by dots; no hash/suffix.
        # Example: qchem_k2_1.3.geom
        fn = os.path.join(out_dir, f"{job_id}.geom")
        with open(fn, "w", encoding="utf-8") as f:
            f.write(geom + "\n")
        count += 1

    typer.echo(f"Generated {count} geometries into: {out_dir}")


@app.command("gen_from_monomer")
def gen_from_monomer(
    monomers_dir: str = typer.Argument(..., help="Directory of monomer .geom files"),
    out_dir: str = typer.Option("mbe_geoms", help="Output directory"),
    max_order: int = typer.Option(2, help="Generate subsets up to this order"),
    orders: Optional[List[int]] = typer.Option(None, "--order", "--orders", help="Explicit subset orders (repeatable)"),
    cp: bool = typer.Option(True, help="Use CP-style ghosts for fragments not in subset"),
    scheme: str = typer.Option("mbe", help="MBE scheme/type label"),
    backend: str = typer.Option("qchem", help="Backend formatting: qchem/orca"),
    monomer_glob: str = typer.Option("*.geom", help="Glob to select monomer .geom files"),
    cluster_name: Optional[str] = typer.Option(None, help="Use as filename prefix (fallback to backend if omitted)"),
):
    """Generate subset geometries directly from existing monomer .geom files."""
    from .mbe import MBEParams, generate_subsets_xyz

    root = Path(monomers_dir)
    if not root.is_dir():
        raise typer.BadParameter(f"monomers_dir must be a directory: {monomers_dir}")

    frags = _load_fragments_from_monomer_dir(root, monomer_glob)
    os.makedirs(out_dir, exist_ok=True)

    params = MBEParams(
        max_order=max_order,
        orders=orders,
        cp_correction=cp,
        backend=backend,
        name_prefix=cluster_name or None,
        scheme=scheme,
    )

    count = 0
    for job_id, subset, geom in generate_subsets_xyz(frags, params):
        fn = os.path.join(out_dir, f"{job_id}.geom")
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
    local_run: bool = typer.Option(
        False,
        "--local-run",
        help="For PBS/Q-Chem: emit a local bash runner instead of submitting via qsub",
    ),
    control_file: Optional[str] = typer.Option(
        None,
        "--control-file",
        help="Optional TOML control file path to enforce during run_with_control",
    ),
    builtin_control: bool = typer.Option(
        False,
        "--builtin-control",
        help="Use the packaged default control policy (writes .mbe_default.control.toml)",
    ),
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
            module=module or "qchem/6.2.2",
            chunk_size=chunk_size,
            local_run=local_run,
            control_file=control_file,
            builtin_control=builtin_control,
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
    charge: Optional[int] = typer.Option(None, help="Total charge (default: auto=0)", show_default=False),
    multiplicity: Optional[int] = typer.Option(None, help="Spin multiplicity (default: auto=1)", show_default=False),
    thresh: Optional[float] = typer.Option(None, help="Q-Chem: THRESH"),
    tole: Optional[float] = typer.Option(None, help="Q-Chem: TolE"),
    scf_convergence: Optional[str] = typer.Option(None, help="SCF convergence keyword (qchem: scf_convergence; orca: TightSCF etc.)"),
    xc_grid: Optional[str] = typer.Option(
        None,
        "--xc-grid",
        help="Q-Chem: XC_GRID value (e.g., 3 or 000075)",
    ),
    grid: Optional[str] = typer.Option(None, help="ORCA grid keyword (e.g., GRID5)"),
    rem_extra: Optional[str] = typer.Option(None, help="Extra Q-Chem $rem lines (newline-separated)"),
    keyword_line_extra: Optional[str] = typer.Option(None, help="Extra ORCA header keywords"),
    out: str = typer.Option("job.inp", help="Output input file"),
    glob_pattern: Optional[str] = typer.Option(None, "--glob", help="Batch mode: build inputs for all matching geom files in a directory"),
    out_dir: Optional[str] = typer.Option(None, help="Batch mode: output directory (defaults to geom directory)"),
):
    """Build a full input file from a geometry block."""
    from .input_builder import build_input_from_geom

    chg = 0 if charge is None else charge
    mult = 1 if multiplicity is None else multiplicity

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
                charge=chg,
                multiplicity=mult,
                thresh=thresh,
                tole=tole,
                scf_convergence=scf_convergence,
                xc_grid=xc_grid,
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
        charge=chg,
        multiplicity=mult,
        thresh=thresh,
        tole=tole,
        scf_convergence=scf_convergence,
        xc_grid=xc_grid,
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

    # Normalize subset_indices for strict calculations (fill missing singleton indices from filenames)
    recs_norm: List[dict] = []
    for r in recs:
        r2 = dict(r)
        if r2.get("subset_indices") is None and r2.get("subset_size") == 1:
            import re

            job_id = r2.get("job_id") or ""
            path = r2.get("path") or ""
            m = re.search(r"k1[_\-]([0-9]+)", job_id) or re.search(r"k1[_\-]([0-9]+)", path)
            if m:
                idx = max(int(m.group(1)) - 1, 0)
                r2["subset_indices"] = [idx]
        recs_norm.append(r2)

    # CPU
    cpu_ok = sum(r.get("cpu_seconds", 0.0) or 0.0 for r in recs_norm if r.get("status") in ("ok", None))
    cpu_all = sum(r.get("cpu_seconds", 0.0) or 0.0 for r in recs_norm)
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

    # Coverage with expected counts when n_monomers is known
    n_monomers = cluster.get("n_monomers") if cluster else None
    coverage: Dict[int, Dict[str, int]] = {}
    for r in recs:
        k = r.get("subset_size")
        if k is None:
            continue
        st = r.get("status") or "ok"
        coverage.setdefault(int(k), {}).setdefault(st, 0)
        coverage[int(k)][st] += 1
    if coverage:
        typer.echo("Coverage:")
        for k in sorted(coverage):
            st_map = coverage[k]
            ok_cnt = st_map.get("ok", 0)
            fail_cnt = sum(v for s, v in st_map.items() if s != "ok")
            parts = [f"ok:{ok_cnt}", f"fail:{fail_cnt}"]
            if n_monomers:
                expected = math.comb(int(n_monomers), int(k)) if int(k) <= int(n_monomers) else 0
                missing = max(expected - ok_cnt - fail_cnt, 0)
                parts.append(f"expected:{expected}")
                parts.append(f"missing:{missing}")
            typer.echo(f"  k={k}: {', '.join(parts)}")

    # MBE strict energies with deltas
    try:
        from .analysis import strict_mbe_orders

        mbe_rows, missing = strict_mbe_orders(recs_norm)
        if mbe_rows:
            typer.echo("MBE (strict inclusion–exclusion):")
            prev = None
            for r in mbe_rows:
                e_val = r.get("energy_hartree", r.get("mbe_energy_hartree"))
                if e_val is None:
                    continue
                delta = (e_val - prev) if prev is not None else None
                if delta is None:
                    typer.echo(f"  order={r['order']} E={e_val:.10f} hartree")
                else:
                    typer.echo(f"  order={r['order']} E={e_val:.10f} hartree ΔE={delta:.10f} hartree")
                prev = e_val
        if missing:
            typer.echo(f"Missing lower-order subsets: {sorted(set(missing))}")
    except Exception as e:
        typer.echo(f"Warning: failed to compute strict MBE energies ({e}). Continuing without this section.", err=True)

    # Monomer participation summary
    if monomer is not None:
        contain: List[dict] = []
        for r in recs:
            indices = r.get("subset_indices")
            if isinstance(indices, list) and monomer in indices:
                contain.append(r)
        if contain:
            cpu_contain = sum(r.get("cpu_seconds", 0.0) or 0.0 for r in contain)
            by_k: Dict[int, int] = {}
            for r in contain:
                if r.get("subset_size") is not None:
                    by_k[int(r["subset_size"])] = by_k.get(int(r["subset_size"]), 0) + 1
            parts = ", ".join(f"k={k}:n={cnt}" for k, cnt in sorted(by_k.items()))
            typer.echo(f"Monomer {monomer} in subsets: jobs={len(contain)} cpu={cpu_contain:.2f}s ({parts})")


@app.command()
def info(
    jsonl_path: Optional[str] = typer.Argument(None, help="JSONL path (defaults apply)"),
    program: Optional[str] = typer.Option(None, help="Filter by program"),
    method: Optional[str] = typer.Option(None, help="Filter by method"),
    basis: Optional[str] = typer.Option(None, help="Filter by basis"),
    grid: Optional[str] = typer.Option(None, help="Filter by grid"),
    cp: Optional[bool] = typer.Option(None, help="Filter by CP correction (true/false)"),
    status: Optional[str] = typer.Option(None, help="Filter by status"),
    scheme: Optional[str] = typer.Option(None, help="MBE summary: simple|strict"),
    max_order: Optional[int] = typer.Option(None, help="Maximum order for MBE summary"),
    json_out: bool = typer.Option(False, "--json", help="Output JSON summary"),
):
    """Summary panel of coverage, CPU, combinations, energies."""
    from .jsonl_selector import select_jsonl
    from .analysis import strict_mbe_orders

    path = select_jsonl(jsonl_path, echo=typer.echo)
    cluster, recs = _load_jsonl_with_cluster(path)

    def _matches(rec: dict) -> bool:
        if program and rec.get("program") != program:
            return False
        if method and rec.get("method") != method:
            return False
        if basis and rec.get("basis") != basis:
            return False
        if grid and rec.get("grid") != grid:
            return False
        if cp is not None and rec.get("cp_correction") is not cp:
            return False
        if status and (rec.get("status") or "ok") != status:
            return False
        return True

    recs = [r for r in recs if _matches(r)]
    if not recs:
        typer.echo("No records after filters")
        raise typer.Exit(code=1)

    if cluster:
        typer.echo(f"Cluster: id={cluster.get('cluster_id')} n_monomers={cluster.get('n_monomers')} (geometry_incomplete={cluster.get('geometry_incomplete')})")
    else:
        typer.echo("Cluster: not available (no cluster record)")

    combos: Dict[Tuple[Any, Any, Any, Any, Any], int] = {}
    by_order_status: Dict[int, Dict[str, int]] = {}
    energy_by_order: Dict[int, list[float]] = {}
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
        if k is not None and r.get("energy_hartree") is not None:
            energy_by_order.setdefault(int(k), []).append(float(r["energy_hartree"]))

    typer.echo("Combinations:")
    for combo, cnt in combos.items():
        typer.echo(f"  {_combo_label(combo)}: {cnt} records")

    n_monomers = cluster.get("n_monomers") if cluster else None
    typer.echo("Coverage by subset_size (status counts):")
    for k in sorted(by_order_status):
        st_map = by_order_status[k]
        ok_cnt = st_map.get("ok", 0)
        fail_cnt = sum(v for s, v in st_map.items() if s != "ok")
        parts = [f"ok:{ok_cnt}", f"fail:{fail_cnt}"]
        if n_monomers:
            expected = math.comb(int(n_monomers), int(k)) if int(k) <= int(n_monomers) else 0
            missing = max(expected - ok_cnt - fail_cnt, 0)
            parts.append(f"expected:{expected}")
            parts.append(f"missing:{missing}")
        typer.echo(f"  k={k}: {', '.join(parts)}")

    typer.echo(f"CPU: ok={cpu_ok:.2f}s all={cpu_all:.2f}s")

    mbe_result = None
    mbe_missing = None
    scheme_l = scheme.lower() if scheme else None
    if scheme_l in {"simple", "strict"}:
        if scheme_l == "strict":
            mbe_rows, missing = strict_mbe_orders(recs, max_order=max_order)
            mbe_result = mbe_rows
            mbe_missing = missing
            if mbe_rows:
                typer.echo("MBE (strict inclusion–exclusion):")
                for r in mbe_rows:
                    e_val = r.get("energy_hartree", r.get("mbe_energy_hartree"))
                    if e_val is None:
                        continue
                    typer.echo(f"  order={r['order']} E={e_val:.10f} hartree")
            if missing:
                typer.echo(f"Missing lower-order subsets: {sorted(set(missing))}")
        else:
            typer.echo("Energy by subset_size (mean/min/max):")
            for k in sorted(energy_by_order):
                vals = energy_by_order[k]
                typer.echo(f"  k={k}: mean={sum(vals)/len(vals):.10f} min={min(vals):.10f} max={max(vals):.10f} hartree")

    if json_out:
        summary = {
            "cluster": cluster,
            "filters": {
                "program": program,
                "method": method,
                "basis": basis,
                "grid": grid,
                "cp_correction": cp,
                "status": status,
            },
            "combos": [{"label": _combo_label(k), "count": v} for k, v in combos.items()],
            "coverage": {str(k): v for k, v in by_order_status.items()},
            "cpu": {"ok": cpu_ok, "all": cpu_all},
            "energy_by_order": {str(k): {"mean": sum(vals)/len(vals), "min": min(vals), "max": max(vals)} for k, vals in energy_by_order.items()},
        }
        if mbe_result is not None:
            summary["mbe"] = mbe_result
        if mbe_missing:
            summary["mbe_missing"] = mbe_missing
        typer.echo(json.dumps(summary, indent=2))
        return


@app.command()
def calc(
    jsonl_path: Optional[str] = typer.Argument(None, help="JSONL path (defaults apply)"),
    scheme: str = typer.Option("simple", help="simple|strict"),
    to: Optional[int] = typer.Option(None, help="Compute up to order K"),
    from_order: Optional[int] = typer.Option(None, "--from", help="Lower order for ΔE(i→j)"),
    monomer: Optional[int] = typer.Option(None, help="Report monomer energy (subset_size=1, index)"),
    unit: str = typer.Option("hartree", help="hartree|kcal|kj"),
    interaction: List[str] = typer.Option(None, "--interaction", help="Subset interaction energy; format i,j or i,j,k"),
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
    combos_list = []
    for r in recs:
        combo = _extract_combo(r)
        combos_list.append(combo)
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
        uniq = sorted({_combo_label(c) for c in combos_list})
        typer.echo("Mixed program/method/basis/grid/cp combinations detected; please split files or filter:")
        for c in uniq:
            typer.echo(f"  {c}")
        raise typer.Exit(code=1)

    # Normalize subset_indices for calculations (fill missing singleton indices from filenames)
    recs_norm: List[dict] = []
    for r in recs:
        r2 = dict(r)
        if r2.get("subset_indices") is None and r2.get("subset_size") == 1:
            import re

            job_id = r2.get("job_id") or ""
            path = r2.get("path") or ""
            m = re.search(r"k1[_\-]([0-9]+)", job_id) or re.search(r"k1[_\-]([0-9]+)", path)
            if m:
                idx = max(int(m.group(1)) - 1, 0)
                r2["subset_indices"] = [idx]
        recs_norm.append(r2)

    # CPU
    cpu_ok = sum(r.get("cpu_seconds", 0.0) or 0.0 for r in recs_norm if r.get("status") in ("ok", None))
    cpu_all = sum(r.get("cpu_seconds", 0.0) or 0.0 for r in recs_norm)
    typer.echo(f"CPU: ok={cpu_ok:.2f}s all={cpu_all:.2f}s")

    # Monomer energy
    if monomer is not None:
        mono = [r for r in recs_norm if r.get("subset_size") == 1 and r.get("subset_indices") == [monomer] and r.get("energy_hartree") is not None]
        if mono:
            e = mono[0]["energy_hartree"]
            typer.echo(f"E(monomer {monomer}) = {e * unit_factor:.10f} {unit_l}")
        else:
            typer.echo(f"Monomer {monomer} energy not found")

    # Interaction energies for specified subsets
    if interaction:
        mono_energy: Dict[int, float] = {}
        for r in recs_norm:
            if r.get("subset_size") == 1 and r.get("subset_indices") and r.get("energy_hartree") is not None:
                idx = r["subset_indices"][0]
                mono_energy[idx] = float(r["energy_hartree"])

        def _parse_subset(tok: str) -> List[int]:
            toks = tok.replace("[", "").replace("]", "").split(",")
            nums: List[int] = []
            for t in toks:
                if not t.strip():
                    continue
                nums.append(int(t.strip()))
            return sorted(nums)

        typer.echo("Interactions (subset energy and Δ vs monomers):")
        for tok in interaction:
            subset = _parse_subset(tok)
            label = ",".join(str(i) for i in subset) if subset else tok
            rec = next((r for r in recs_norm if r.get("subset_indices") == subset and r.get("energy_hartree") is not None), None)
            if rec is None:
                typer.echo(f"  [{label}]: subset not found")
                continue
            e_h = float(rec["energy_hartree"])
            delta_h = None
            if subset and all(i in mono_energy for i in subset):
                delta_h = e_h - sum(mono_energy[i] for i in subset)
            e_u = e_h * unit_factor
            if delta_h is None:
                typer.echo(f"  [{label}]: E={e_u:.10f} {unit_l} (no monomer reference)")
            else:
                typer.echo(f"  [{label}]: E={e_u:.10f} {unit_l}; ΔE vs singles={delta_h * unit_factor:.10f} {unit_l}")

    # Energy aggregation
    if scheme.lower() == "strict":
        rows, missing = strict_mbe_orders(recs_norm, max_order=to)
        if rows:
            typer.echo("MBE (strict inclusion–exclusion):")
            for r in rows:
                e_val = r.get("energy_hartree", r.get("mbe_energy_hartree"))
                if e_val is None:
                    continue
                typer.echo(f"  order={r['order']} E={e_val * unit_factor:.10f} {unit_l}")
        if missing:
            typer.echo(f"Missing lower-order subsets: {sorted(set(missing))}")
    else:
        # simple: mean by order and optional ΔE(from→to); also print reference strict MBE table if --to provided
        by_order: Dict[int, List[float]] = {}
        for r in recs_norm:
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
            if to is not None:
                rows, missing = strict_mbe_orders(recs_norm, max_order=to)
                if rows:
                    typer.echo("MBE (strict inclusion–exclusion, reference):")
                    for r in rows:
                        e_val = r.get("energy_hartree", r.get("mbe_energy_hartree"))
                        if e_val is None:
                            continue
                        typer.echo(f"  order={r['order']} E={e_val * unit_factor:.10f} {unit_l}")
                if missing:
                    typer.echo(f"Missing lower-order subsets: {sorted(set(missing))}")


@app.command()
def save(
    jsonl_path: Optional[str] = typer.Argument(None, help="JSONL path (defaults apply)"),
    dest: Optional[str] = typer.Option(None, help="Destination directory (default: $MBE_SAVE_DEST or ~/.local/share/mbe-tools/runs)", show_default=False),
    include_energy: bool = typer.Option(True, help="Include MBE(order) energy in meta when available"),
    order: Optional[int] = typer.Option(None, help="Order for MBE energy (defaults to max computed)"),
):
    """Archive JSONL (copy) to a default run path or an explicit destination."""
    from .jsonl_selector import select_jsonl

    path = select_jsonl(jsonl_path, echo=typer.echo)
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    cluster, recs = _load_jsonl_with_cluster(path)
    cluster_id = (cluster.get("cluster_id") if cluster else None) or Path(path).stem

    combos = {_extract_combo(r) for r in recs}

    def _slug(val: Any, fallback: str) -> str:
        if val is None:
            return fallback
        s = str(val).strip()
        return s.replace("/", "-").replace("\\", "-") or fallback

    if len(combos) == 1:
        combo = next(iter(combos))
        method_label = _slug(combo[1], "method?")
        basis_label = _slug(combo[2], "basis?")
        grid_label = _slug(combo[3], "grid?")
        cp_label = "cp" if combo[4] is True else ("nocp" if combo[4] is False else "cp?")
    else:
        method_label = "mixed"
        basis_label = "mixed"
        grid_label = "mixed"
        cp_label = "mixed"

    dest_root = _resolve_library_root(dest)
    dest_dir = dest_root / cluster_id / f"{stamp}__{method_label}__{basis_label}__{grid_label}__{cp_label}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "run.jsonl"
    shutil.copy2(path, dest_path)

    # Emit a small meta file for quick browsing
    combos_labels = sorted({_combo_label(_extract_combo(r)) for r in recs})
    cpu_ok = sum(r.get("cpu_seconds", 0.0) or 0.0 for r in recs if r.get("status") in ("ok", None))
    cpu_all = sum(r.get("cpu_seconds", 0.0) or 0.0 for r in recs)
    coverage: Dict[str, Dict[str, int]] = {}
    n_monomers = cluster.get("n_monomers") if cluster else None
    for r in recs:
        k = r.get("subset_size")
        if k is None:
            continue
        k_s = str(int(k))
        st = r.get("status") or "ok"
        coverage.setdefault(k_s, {}).setdefault(st, 0)
        coverage[k_s][st] += 1
    if n_monomers:
        for k_s, st_map in coverage.items():
            k_int = int(k_s)
            expected = math.comb(int(n_monomers), k_int) if k_int <= int(n_monomers) else 0
            ok_cnt = st_map.get("ok", 0)
            fail_cnt = sum(v for s, v in st_map.items() if s != "ok")
            st_map["expected"] = expected
            st_map["missing"] = max(expected - ok_cnt - fail_cnt, 0)

    mbe_energy = None
    mbe_order = None
    if include_energy:
        try:
            from .analysis import strict_mbe_orders

            rows, missing = strict_mbe_orders(recs, max_order=order)
            if rows:
                pick = order if order is not None else rows[-1]["order"]
                for r in rows:
                    if r.get("order") == pick:
                        mbe_energy = r.get("energy_hartree", r.get("mbe_energy_hartree"))
                        mbe_order = pick
                        break
                if missing:
                    coverage["missing_subsets"] = missing
        except Exception:
            mbe_energy = None

    meta = {
        "source": str(Path(path).resolve()),
        "saved_at_utc": stamp,
        "cluster_id": cluster_id,
        "n_monomers": n_monomers,
        "combos": combos_labels,
        "cpu_ok": cpu_ok,
        "cpu_all": cpu_all,
        "coverage": coverage,
        "mbe_energy_hartree": mbe_energy,
        "mbe_order": mbe_order,
    }
    meta_path = dest_dir / "run.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    typer.echo(f"Saved to {dest_path}\nMeta: {meta_path}")


@app.command()
def compare(
    paths_or_globs: List[str] = typer.Argument(..., help="One or more paths/dirs/globs for *.jsonl"),
    cluster: Optional[str] = typer.Option(None, help="Filter cluster_id"),
    scheme: str = typer.Option("strict", help="Energy aggregation: simple|strict"),
    order: Optional[int] = typer.Option(None, help="Order for energy comparison (MBE order or subset_size)"),
    ref: str = typer.Option("latest", help="Reference run: latest|first|PATH"),
    to_csv: Optional[str] = typer.Option(None, help="Export comparison table to CSV"),
    to_xlsx: Optional[str] = typer.Option(None, help="Export comparison table to XLSX (requires pandas)"),
    output_format: str = typer.Option("pipe", help="Output format: pipe|tsv|table", show_default=True),
    show_combos: bool = typer.Option(False, help="Include combos column in stdout output"),
    show_coverage: bool = typer.Option(False, help="Include coverage column in stdout output"),
):
    """Compare multiple JSONL runs (CPU + counts)."""
    import glob as _glob

    paths: List[str] = []
    for arg in paths_or_globs:
        p = Path(arg)
        if p.is_dir():
            paths.extend(str(x) for x in p.rglob("*.jsonl"))
        else:
            paths.extend(_glob.glob(arg, recursive=True))
    paths = sorted({str(Path(p).resolve()) for p in paths})
    if not paths:
        raise typer.BadParameter("No JSONL files found for compare")

    def _summarize(path: str) -> Dict[str, Any]:
        cl, recs = _load_jsonl_with_cluster(path)
        cid = cl.get("cluster_id") if cl else "unknown"
        # Normalize subset_indices for singleton records (mirror calc/show behavior)
        recs_norm: List[dict] = []
        for r in recs:
            r2 = dict(r)
            if r2.get("subset_indices") is None and r2.get("subset_size") == 1:
                import re

                job_id = r2.get("job_id") or ""
                pth = r2.get("path") or ""
                m = re.search(r"k1[_\-]([0-9]+)", job_id) or re.search(r"k1[_\-]([0-9]+)", pth)
                if m:
                    idx = max(int(m.group(1)) - 1, 0)
                    r2["subset_indices"] = [idx]
            recs_norm.append(r2)

        cpu_ok = sum(r.get("cpu_seconds", 0.0) or 0.0 for r in recs if r.get("status") in ("ok", None))
        counts = len(recs)
        combos = {_combo_label(_extract_combo(r)) for r in recs}
        coverage: Dict[int, Dict[str, int]] = {}
        energies: Dict[int, List[float]] = {}
        for r in recs_norm:
            k = r.get("subset_size")
            e = r.get("energy_hartree")
            st = r.get("status") or "ok"
            if k is not None:
                coverage.setdefault(int(k), {}).setdefault(st, 0)
                coverage[int(k)][st] += 1
            if k is not None and e is not None:
                energies.setdefault(int(k), []).append(float(e))

        energy_val = None
        energy_desc = None
        scheme_l = scheme.lower()
        if scheme_l == "strict":
            from .analysis import strict_mbe_orders

            rows, missing = strict_mbe_orders(recs_norm, max_order=order)
            pick_order = order if order is not None else (rows[-1]["order"] if rows else None)
            if rows and pick_order is not None:
                for r in rows:
                    if r.get("order") == pick_order:
                        energy_val = r.get("energy_hartree", r.get("mbe_energy_hartree"))
                        break
            if energy_val is not None:
                energy_desc = f"MBE(order={pick_order})"
            if missing:
                energy_desc = (energy_desc + "; missing=" if energy_desc else "missing=") + str(sorted(set(missing)))
        else:
            k = order if order is not None else (max(energies.keys()) if energies else None)
            if k is not None and k in energies:
                vals = energies[k]
                energy_val = sum(vals) / len(vals)
                energy_desc = f"mean(k={k})"

        cov_txt = "; ".join(
            f"k={k}:" + ",".join(f"{s}:{n}" for s, n in coverage[k].items()) for k in sorted(coverage)
        )
        ene_txt = "; ".join(
            f"k={k}:mean={sum(v)/len(v):.6f}" for k, v in sorted(energies.items())
        )
        return {
            "cluster_id": cid,
            "file": Path(path).name,
            "path": str(Path(path).resolve()),
            "mtime": Path(path).stat().st_mtime,
            "cpu_ok": cpu_ok,
            "records": counts,
            "combos": ",".join(sorted(combos))[:120],
            "coverage": cov_txt,
            "energy_means": ene_txt,
            "energy_value": energy_val,
            "energy_desc": energy_desc,
        }

    summaries: List[Dict[str, Any]] = []
    for path in sorted(paths):
        summ = _summarize(path)
        if cluster and summ["cluster_id"] != cluster:
            continue
        summaries.append(summ)

    if not summaries:
        raise typer.BadParameter("No matching JSONL after filters")

    def _select_ref() -> Dict[str, Any]:
        ref_l = ref.lower()
        if ref_l == "latest":
            return max(summaries, key=lambda r: r["mtime"])
        if ref_l == "first":
            return min(summaries, key=lambda r: r["mtime"])
        # explicit path
        target = Path(ref).resolve()
        for r in summaries:
            if Path(r["path"]) == target:
                return r
        # if not in summaries, try loading directly (must match cluster if provided)
        extra = _summarize(str(target))
        if cluster and extra["cluster_id"] != cluster:
            raise typer.BadParameter("Reference file cluster_id does not match filter")
        summaries.append(extra)
        return extra

    ref_row = _select_ref()
    unique_paths = {Path(r["path"]).resolve() for r in summaries}
    if len(unique_paths) < 2:
        raise typer.BadParameter("compare requires at least two distinct JSONL files; provide multiple paths or a glob that matches 2+")
    ref_energy = ref_row.get("energy_value")
    ref_cpu = ref_row.get("cpu_ok", 0.0)

    rows: List[Dict[str, Any]] = []
    for r in summaries:
        rows.append(
            {
                "ref": "*" if r is ref_row else "",
                "cluster_id": r["cluster_id"],
                "file": r["file"],
                "cpu_ok": r["cpu_ok"],
                "cpu_delta": (r["cpu_ok"] - ref_cpu) if ref_cpu is not None else None,
                "records": r["records"],
                "combos": r["combos"],
                "coverage": r["coverage"],
                "energy": r.get("energy_value"),
                "energy_delta": (r.get("energy_value") - ref_energy) if (ref_energy is not None and r.get("energy_value") is not None) else None,
                "energy_desc": r.get("energy_desc"),
            }
        )

    # Nicely aligned table to avoid pipes inside data breaking columns
    headers = [
        "ref",
        "cluster_id",
        "file",
        "cpu_ok(s)",
        "Δcpu(s)",
        "records",
        "energy",
        "Δenergy",
        "desc",
    ]
    if show_combos:
        headers.insert(6, "combos")
    if show_coverage:
        insert_at = 7 if show_combos else 6
        headers.insert(insert_at, "coverage")

    def fmt_row(r: Dict[str, Any]) -> Dict[str, str]:
        return {
            "ref": r.get("ref") or " ",
            "cluster_id": str(r.get("cluster_id", "")),
            "file": str(r.get("file", "")),
            "cpu_ok(s)": f"{r.get('cpu_ok', float('nan')):.2f}",
            "Δcpu(s)": f"{(r.get('cpu_delta') if r.get('cpu_delta') is not None else float('nan')):.2f}",
            "records": str(r.get("records", "")),
            "combos": str(r.get("combos", "")),
            "coverage": str(r.get("coverage", "")),
            "energy": f"{(r.get('energy') if r.get('energy') is not None else float('nan')):.6f}",
            "Δenergy": f"{(r.get('energy_delta') if r.get('energy_delta') is not None else float('nan')):.6f}",
            "desc": str(r.get("energy_desc", "")),
        }

    display_rows = [fmt_row(r) for r in rows]

    fmt = output_format.lower()
    if fmt not in {"pipe", "tsv", "table"}:
        raise typer.BadParameter("--output-format must be pipe, tsv, or table")

    if fmt == "tsv":
        typer.echo("\t".join(headers))
        for r in display_rows:
            typer.echo("\t".join(r[h] for h in headers))
    else:
        # sanitize internal separators
        for r in display_rows:
            r["combos"] = r["combos"].replace("|", "/")
            r["coverage"] = r["coverage"].replace("|", "/")
            for k in r:
                r[k] = r[k].replace("\t", " ")

        widths: Dict[str, int] = {h: len(h) for h in headers}
        for r in display_rows:
            for h in headers:
                widths[h] = max(widths[h], len(r[h]))

        def render(r: Dict[str, str]) -> str:
            return " | ".join(r[h].ljust(widths[h]) for h in headers)

        typer.echo(render({h: h for h in headers}))
        for r in display_rows:
            typer.echo(render(r))

    if to_csv:
        out_p = Path(to_csv)
        with out_p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        typer.echo(f"Wrote CSV: {out_p}")

    if to_xlsx:
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:
            raise typer.BadParameter(f"pandas required for --to-xlsx ({exc})")
        df = pd.DataFrame(rows)
        df.to_excel(to_xlsx, index=False)
        typer.echo(f"Wrote XLSX: {to_xlsx}")


@app.command()
def where():
    """Print default data/config/cache/state paths."""
    paths = _xdg_paths()
    runs_path = paths["data"] / "runs"
    typer.echo(f"data:  {paths['data']}")
    typer.echo(f"config:{paths['config']}")
    typer.echo(f"cache: {paths['cache']}")
    typer.echo(f"state: {paths['state']}")
    typer.echo(f"runs:  {runs_path}")


@app.command(name="set-library")
def set_library(path: str = typer.Argument(..., help="Directory to store archived runs (mbe save default)")):
    """Persist default library path for mbe save (overridden by --dest or MBE_SAVE_DEST)."""

    target = Path(path).expanduser().resolve()
    cfg = _library_config_path()
    try:
        cfg.write_text(str(target), encoding="utf-8")
    except OSError as exc:
        raise typer.BadParameter(f"Failed to write library path: {exc}")
    typer.echo(f"Default library set to: {target}\nStored in: {cfg}")


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

    # Fill missing subset_indices for singleton jobs using filename hints (matches calc/show behavior)
    import re

    for rec in recs:
        if (rec.subset_indices is None or rec.subset_indices == []) and rec.subset_size in (None, 1):
            job_id = rec.job_id or ""
            pth = rec.path or ""
            m = re.search(r"k1[_\-]([0-9]+)", job_id) or re.search(r"k1[_\-]([0-9]+)", pth)
            if m:
                idx = max(int(m.group(1)) - 1, 0)
                rec.subset_indices = [idx]

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
            if not parts:
                i += 1
                if atoms:
                    break
                continue
            if set(parts[0]) <= {"-", "="}:
                i += 1
                continue
            if len(parts) < 4:
                i += 1
                if atoms:
                    break
                continue
            # handle optional leading index
            if parts[0].isdigit():
                parts = parts[1:]
            if len(parts) < 4:
                break
            el = parts[0]
            try:
                x, y, z = map(float, parts[1:4])
            except Exception:
                # skip header/garbage lines and continue searching
                i += 1
                continue
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
            cluster_id_val = Path(root).name or Path(root).resolve().name
            cluster_record = {
                "record_type": "cluster",
                "schema_version": 2,
                "cluster_id": cluster_id_val,
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
def enrich(
    jsonl_path: str = typer.Argument(..., help="Existing JSONL (calc-only) to enrich with geometry"),
    root: str = typer.Option(".", help="Root directory containing output files referenced in JSONL"),
    program: str = typer.Option("auto", help="qchem/orca/auto"),
    geom_max_lines: int = typer.Option(5000, help="Scan first N lines for geometry blocks"),
    geom_mode: str = typer.Option("first", help="Geometry block pick: first|last"),
    geom_source: str = typer.Option("singleton", help="Geometry extraction source: singleton|any"),
    geom_drop_ghost: bool = typer.Option(True, help="Drop ghost atoms when extracting geometry"),
    out: Optional[str] = typer.Option(None, help="Output JSONL (default: <input>.enriched.jsonl)"),
):
    """Embed a cluster geometry record into an existing JSONL."""
    from .analysis import read_jsonl

    rows = read_jsonl(jsonl_path)
    existing_cluster = next((r for r in rows if isinstance(r, dict) and r.get("record_type") == "cluster"), None)
    if existing_cluster:
        typer.echo("Cluster record already present; nothing to do.")
        raise typer.Exit()

    import re

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
            if not parts:
                i += 1
                if atoms:
                    break
                continue
            if set(parts[0]) <= {"-", "="}:
                i += 1
                continue
            if len(parts) < 4:
                i += 1
                if atoms:
                    break
                continue
            if parts[0].isdigit():
                parts = parts[1:]
            if len(parts) < 4:
                i += 1
                if atoms:
                    break
                continue
            el = parts[0]
            try:
                x, y, z = map(float, parts[1:4])
            except Exception:
                i += 1
                continue
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
                    j = idx + 1
                    while j < len(lines) and (not lines[j].strip() or set(lines[j].strip()) <= {"-", "="}):
                        j += 1
                    atoms, end_idx = _parse_block(lines, j)
                    if atoms:
                        matches.append((idx, atoms))
            elif prog_l == "orca":
                if "cartesian coordinates (angstrom" in ln_l:
                    j = idx + 1
                    while j < len(lines) and lines[j].strip():
                        j += 1
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

    # Normalize subset_indices for singleton records
    calc_rows: list[dict] = []
    for r in rows:
        if isinstance(r, dict) and r.get("record_type") == "cluster":
            continue
        r2 = dict(r)
        if (r2.get("subset_indices") is None or r2.get("subset_indices") == []) and r2.get("subset_size") in (None, 1):
            job_id = r2.get("job_id") or ""
            pth = r2.get("path") or ""
            m = re.search(r"k1[_\-]([0-9]+)", job_id) or re.search(r"k1[_\-]([0-9]+)", pth)
            if m:
                idx = max(int(m.group(1)) - 1, 0)
                r2["subset_indices"] = [idx]
                r2["subset_size"] = 1
        calc_rows.append(r2)

    geom_source_mode = geom_source.lower()
    if geom_source_mode not in {"singleton", "any"}:
        raise typer.BadParameter("--geom-source must be singleton or any")

    def _eligible_records() -> list[dict]:
        eligible: list[dict] = []
        for r in calc_rows:
            if r.get("status") not in ("ok", None):
                continue
            indices = r.get("subset_indices")
            if not (isinstance(indices, list) and len(indices) == 1):
                continue
            if geom_source_mode == "singleton" and r.get("subset_size") not in (None, 1):
                continue
            eligible.append(r)
        return eligible

    eligible = _eligible_records()
    geom_by_idx: dict[int, list[Atom]] = {}
    for r in eligible:
        m_idx = r["subset_indices"][0]
        path = r.get("path") or ""
        full_path = path if os.path.isabs(path) else os.path.join(root, path)
        prog_val = r.get("program") or program
        atoms = extract_geometry_from_out_head(full_path, prog_val)
        if atoms:
            geom_by_idx.setdefault(m_idx, atoms)

    if not geom_by_idx:
        typer.echo("Geometry not found (no cluster record written)")
        raise typer.Exit(code=1)

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

    cluster_id_val = Path(root).name or Path(root).resolve().name
    cluster_record = {
        "record_type": "cluster",
        "schema_version": 2,
        "cluster_id": cluster_id_val,
        "source": {"type": "out_enrich", "path": str(root)},
        "unit": "angstrom",
        "n_monomers": n_monomers,
        "monomers": monomers,
        "geometry_incomplete": bool(missing),
        "missing_monomers": missing,
        "extra": {},
    }

    out_path = Path(out) if out else Path(jsonl_path).with_name(Path(jsonl_path).stem + ".enriched.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(cluster_record, ensure_ascii=False) + "\n")
        for r in calc_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    typer.echo(f"Enriched JSONL written: {out_path}")
    if missing:
        typer.echo(f"Geometry: embedded but incomplete (missing={missing})")
    else:
        typer.echo("Geometry: embedded (cluster record written)")


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
    # Drop cluster/meta rows to keep calc-only stats
    records = [r for r in records if not (isinstance(r, dict) and r.get("record_type") == "cluster")]
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
