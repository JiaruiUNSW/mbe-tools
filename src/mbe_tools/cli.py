from __future__ import annotations
import json
import os
from typing import Optional, List


def _lazy_import_typer():
    try:
        import typer
        return typer
    except Exception as e:
        raise RuntimeError("CLI requires typer. Install with: pip install mbe-tools[cli]") from e


typer = _lazy_import_typer()
app = typer.Typer(add_completion=False)


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
):
    """Build a full input file from a geometry block."""
    from .input_builder import build_input_from_geom

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


@app.command()
def parse(
    root: str = typer.Argument(..., help="Root directory containing outputs"),
    program: str = typer.Option("qchem", help="qchem/orca/auto"),
    glob_pattern: str = typer.Option("*.out", help="Glob pattern, e.g. '*.out'"),
    out: str = typer.Option("parsed.jsonl", help="Output JSONL file"),
    infer_metadata: bool = typer.Option(True, help="Infer subset/method/basis metadata from paths"),
):
    """Parse output files to JSONL."""
    from .parsers.io import glob_paths, parse_files

    paths = glob_paths(root, glob_pattern)
    recs = parse_files(paths, program=program, infer_metadata=infer_metadata)
    with open(out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")
    typer.echo(f"Parsed {len(recs)} files → {out}")


@app.command()
def analyze(
    jsonl_path: str = typer.Argument(..., help="Input JSONL"),
    to_csv: Optional[str] = typer.Option(None, help="Write full table to CSV"),
    to_xlsx: Optional[str] = typer.Option(None, help="Write full table to Excel"),
    plot: Optional[str] = typer.Option(None, help="Plot delta energy (requires matplotlib)"),
    scheme: str = typer.Option("simple", help="Energy aggregation: simple|strict (inclusion–exclusion)"),
    max_order: Optional[int] = typer.Option(None, help="Maximum order for strict aggregation"),
):
    """Analyze parsed JSONL (basic summaries + exports)."""
    from .analysis import read_jsonl, to_dataframe, summarize_by_order, compute_delta_energy, strict_mbe_orders

    records = read_jsonl(jsonl_path)
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
