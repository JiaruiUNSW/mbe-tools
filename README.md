# mbe-tools

`mbe-tools` is a Python package that covers the full **Many-Body Expansion (MBE)** loop:

- **Cluster design**: read `.xyz`, extract fragments, and sample subsets (with optional ion retention).
- **MBE job prep**: generate subset geometries, build Q-Chem/ORCA inputs, and emit PBS/Slurm job scripts (with optional chunked submission).
- **Parsing**: read ORCA / Q-Chem outputs, infer method/basis/grid metadata from paths or companion inputs, and write JSONL.
- **Analysis**: inclusion–exclusion MBE(k), summaries, CSV/Excel export, and basic plots.

Status: **0.1.3 (MVP)**. Backend syntax (e.g., ghost atoms) may need local tweaks.

**What changed (P0–P5)**
- P0: Settings loader and precedence (`mbe.toml`, env vars) for commands/modules/scratch/queue hints.
- P1: Parser auto-detects program and infers method/basis/grid metadata from paths; emits detection + error tags.
- P2: Spatial fragment sampling mode with centroid-based compact selection and ion retention.
- P3: Connectivity-driven fragmentation and fragment labeling (water/methanol/ethanol/benzene/ions).
- P4: Tunable MBE params and input builders (thresh/tole/scf, grid) exposed in CLI.
- P5: Run-control for PBS/Slurm templates (regex confirmation, retries/cleanup, optional delete, state JSON).

---

## Install (editable for development)

```bash
cd mbe-tools
python -m pip install -e .[analysis,cli]
```

## Global settings (P0)

Configure default commands/modules/scratch once and reuse across CLI calls. Precedence (lowest → highest): environment variables → `~/.config/mbe-tools/config.toml` → `./mbe.toml` → explicit path passed to `load_settings(path=...)`.

Supported keys: `qchem_command`, `orca_command`, `qchem_module`, `orca_module`, `scratch_dir`, `scheduler_queue`, `scheduler_partition`, `scheduler_account`.

Environment variable map: `MBE_QCHEM_CMD`, `MBE_ORCA_CMD`, `MBE_QCHEM_MODULE`, `MBE_ORCA_MODULE`, `MBE_SCRATCH`, `MBE_SCHED_QUEUE`, `MBE_SCHED_PARTITION`, `MBE_SCHED_ACCOUNT`.

Minimal `mbe.toml` example (edit paths/modules to your site):

```toml
qchem_command = "/opt/qchem/bin/qchem"
orca_command  = "/opt/orca/bin/orca"
qchem_module  = "qchem/5.2.2"
orca_module   = "orca/5.0.3"
scratch_dir   = "/scratch/${USER}"
scheduler_queue = "normal"          # PBS
scheduler_partition = "work"        # Slurm
scheduler_account = "proj123"
```

---
# mbe-tools (v0.1.3)

`mbe-tools` is a Python toolkit for the **Many-Body Expansion (MBE)** workflow:

- Cluster handling: read `.xyz`, fragment (water heuristic or connectivity + labels), and sample fragments (random/spatial, ion-aware).
- Job prep: generate subset geometries, render Q-Chem/ORCA inputs, and emit PBS/Slurm scripts (supports chunked submission with run-control).
- Parsing: read ORCA/Q-Chem outputs, auto-detect program, infer method/basis/grid metadata, emit JSONL.
- Analysis: inclusion–exclusion MBE(k), summaries, CSV/Excel export, and quick plots.

Status: **v0.1.3 (MVP)** — backend syntax (e.g., ghost atoms) can be customized per site. License: **MIT**.

---

## Install (editable for development)

```bash
cd mbe-tools
python -m pip install -e .[analysis,cli]
```

## Settings precedence (P0)

Configure default commands/modules/scratch once and reuse across CLI calls. Precedence (low → high):
1) env vars → 2) `~/.config/mbe-tools/config.toml` → 3) `./mbe.toml` → 4) explicit `load_settings(path=...)`.

Keys: `qchem_command`, `orca_command`, `qchem_module`, `orca_module`, `scratch_dir`, `scheduler_queue`, `scheduler_partition`, `scheduler_account`.

Env map: `MBE_QCHEM_CMD`, `MBE_ORCA_CMD`, `MBE_QCHEM_MODULE`, `MBE_ORCA_MODULE`, `MBE_SCRATCH`, `MBE_SCHED_QUEUE`, `MBE_SCHED_PARTITION`, `MBE_SCHED_ACCOUNT`.

Minimal `mbe.toml`:

```toml
qchem_command = "/opt/qchem/bin/qchem"
orca_command  = "/opt/orca/bin/orca"
qchem_module  = "qchem/5.2.2"
orca_module   = "orca/5.0.3"
scratch_dir   = "/scratch/${USER}"
scheduler_queue = "normal"
scheduler_partition = "work"
scheduler_account = "proj123"
```

## Quickstart (Python API)

1) Fragment an XYZ

```python
from mbe_tools.cluster import read_xyz, fragment_by_water_heuristic, fragment_by_connectivity

xyz = read_xyz("Water20.xyz")
frags = fragment_by_water_heuristic(xyz, oh_cutoff=1.25)
frags_conn = fragment_by_connectivity(xyz, scale=1.2)
```

2) Sample and write XYZ

```python
from mbe_tools.cluster import sample_fragments, write_xyz

picked = sample_fragments(frags, n=10, seed=42)
write_xyz("Water10_sample.xyz", picked)
```

3) Generate subset geometries

```python
from mbe_tools.mbe import MBEParams, generate_subsets_xyz

params = MBEParams(max_order=3, cp_correction=True, backend="qchem")
subset_jobs = list(generate_subsets_xyz(frags, params))  # (job_id, subset_indices, geom_text)
```

4) Build inputs

```bash
mbe build-input water.geom --backend qchem --method wb97m-v --basis def2-ma-qzvpp --out water_qchem.inp
mbe build-input water.geom --backend orca  --method wb97m-v --basis def2-ma-qzvpp --out water_orca.inp
```

5) Emit PBS/Slurm templates (run-control included)

```bash
mbe template --scheduler pbs   --backend qchem --job-name mbe-qchem --chunk-size 20 --out qchem.pbs
mbe template --scheduler slurm --backend orca  --job-name mbe-orca  --partition work --chunk-size 10 --out orca.sbatch
```

6) Parse outputs to JSONL

```bash
mbe parse ./Output --program auto --glob "*.out" --out parsed.jsonl
```

7) Analyze JSONL

```bash
mbe analyze parsed.jsonl --to-csv results.csv --to-xlsx results.xlsx --plot mbe.png
```

## CLI cheat sheet

- `mbe fragment <xyz>`: water-heuristic fragmentation + sampling → XYZ. Options: `--out-xyz [sample.xyz]`, `--n [10]`, `--seed`, `--require-ion`, `--mode [random|spatial]`, spatial extras `--prefer-special`, `--k-neighbors`, `--start-index`, `--oh-cutoff`.
- `mbe gen <xyz>`: generate subset geometries. Options: `--out-dir [mbe_geoms]`, `--max-order [2]`, `--order/--orders`, `--cp/--no-cp`, `--scheme`, `--backend [qchem|orca]`, `--oh-cutoff`.
- `mbe build-input <geom>`: render Q-Chem/ORCA input. Options for backend, method, basis (required), charge/multiplicity, Q-Chem (`--thresh`, `--tole`, `--scf-convergence`, `--rem-extra`), ORCA (`--grid`, `--scf-convergence`, `--keyword-line-extra`), `--out`.
- `mbe template`: PBS/Slurm scripts with run-control wrapper. Shared: `--scheduler [pbs|slurm]`, `--backend [qchem|orca]`, `--job-name`, `--walltime`, `--mem-gb`, `--chunk-size`, `--module`, `--command`, `--out`; PBS+qchem adds `--ncpus`, `--queue`, `--project`; Slurm+orca adds `--ncpus` (cpus-per-task), `--ntasks`, `--partition`, `--project` (account), `--qos`.
- `mbe parse <root>`: outputs → JSONL. Options: `--program [auto|qchem|orca]`, `--glob-pattern`, `--out`, `--infer-metadata`.
- `mbe analyze <parsed.jsonl>`: summaries/exports. Options: `--to-csv`, `--to-xlsx`, `--plot`, `--scheme [simple|strict]`, `--max-order`.

Use `mbe <command> --help` for full flags.

## Run-control (templates)

- Control file discovery: prefer `<input>.mbe.control.toml`, else `mbe.control.toml`, else run-control disabled.
- Attempt logging: write `job._try.out`; on failure rename to `job.attemptN.out`; on success rename to `job.out`. `confirm.log_path` can override temp log location.
- Confirmation: `confirm.regex_any` (must match) and `confirm.regex_none` (must not match) on the temp log; success also requires exit code 0.
- Retry: `retry.enabled`, `max_attempts`, `sleep_seconds`, `cleanup_globs`, `write_failed_last` (copy last attempt to `failed_last_path`).
- Delete safeguards: `delete.enabled` + `allow_delete_outputs=true` to delete outputs; inputs removed only if matched by `delete_inputs_globs`.
- State: `.mbe_state.json` records status, attempts, matched regex, log paths; `skip_if_done` skips reruns when marked done.

## Subset naming

- Recommended: `{backend}_k{order}_f{i1}-{i2}-{i3}_{cp|nocp}_{hash}` with **0-based** fragment indices (zero-padding allowed), e.g., `qchem_k2_f000-003_cp_deadbeef.out`.
- Legacy (still parsed): `{backend}_k{order}_{i1}.{i2}..._{hash}` treated as 1-based in the name but converted to 0-based internally.
JSON always exposes `subset_indices` as 0-based.

## JSONL schema (parse output)

```json
{
  "job_id": "qchem_k2_f000-003_cp_deadbeef",
  "program": "qchem",
  "program_detected": "qchem",
  "status": "ok",
  "error_reason": null,
  "path": ".../job.out",
  "energy_hartree": -458.7018184,
  "cpu_seconds": 1234.5,
  "wall_seconds": 1234.5,
  "method": "wB97M-V",
  "basis": "def2-ma-QZVPP",
  "grid": "SG-2",
  "subset_size": 2,
  "subset_indices": [0, 2],
  "cp_correction": true,
  "extra": {}
}
```

## API highlights

- Cluster ([src/mbe_tools/cluster.py](src/mbe_tools/cluster.py)): `read_xyz`, `write_xyz`, `fragment_by_water_heuristic`, `fragment_by_connectivity`, `sample_fragments`, `spatial_sample_fragments`.
- MBE generation ([src/mbe_tools/mbe.py](src/mbe_tools/mbe.py)): `MBEParams`, `generate_subsets_xyz`, `qchem_molecule_block`, `orca_xyz_block`.
- Input builders ([src/mbe_tools/input_builder.py](src/mbe_tools/input_builder.py)): `render_qchem_input`, `render_orca_input`, `build_input_from_geom`.
- HPC templates ([src/mbe_tools/hpc_templates.py](src/mbe_tools/hpc_templates.py)): `render_pbs_qchem`, `render_slurm_orca` (both embed run-control wrapper).
- Parsing ([src/mbe_tools/parsers/io.py](src/mbe_tools/parsers/io.py)): `detect_program`, `parse_files`, `infer_metadata_from_path`, `glob_paths`.
- Analysis ([src/mbe_tools/analysis.py](src/mbe_tools/analysis.py)): `read_jsonl`, `summarize_by_order`, `compute_delta_energy`, `strict_mbe_orders`.

## Notebook

See `notebooks/sample_walkthrough.ipynb` for an end-to-end demo: build inputs, generate templates, and assemble MBE(k) energies from synthetic data.

## License

MIT