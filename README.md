# mbe-tools

`mbe-tools` is a Python package that covers the full **Many-Body Expansion (MBE)** loop:

- **Cluster design**: read `.xyz`, extract fragments, and sample subsets (with optional ion retention).
- **MBE job prep**: generate subset geometries, build Q-Chem/ORCA inputs, and emit PBS/Slurm job scripts (with optional chunked submission).
- **Parsing**: read ORCA / Q-Chem outputs, infer method/basis/grid metadata from paths or companion inputs, and write JSONL.
- **Analysis**: inclusion–exclusion MBE(k), summaries, CSV/Excel export, and basic plots.

Status: **0.2.0-dev**. Backend syntax (e.g., ghost atoms) may need local tweaks.

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

Status: **v0.2.0-dev** — backend syntax (e.g., ghost atoms) can be customized per site. License: **MIT**.

## New in v0.2.0 (CLI)

- Default JSONL selection (explicit → run.jsonl → parsed.jsonl → single → newest) reused by analyze/show/info/calc/save/compare.
- New commands: `mbe show`, `mbe info`, `mbe calc`, `mbe save`, `mbe compare` for quick summaries, energy tables, archiving, and multi-run comparisons.
- Parse geometry embedding: `--cluster-xyz` to force geometry, or search outputs with `--geom-mode`, `--geom-source`, `--geom-drop-ghost`, `--nosearch`.

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
- `mbe build-input <geom>`: render Q-Chem/ORCA input. Options for backend, method, basis (required), charge/multiplicity, Q-Chem (`--thresh`, `--tole`, `--scf-convergence`, `--rem-extra`), ORCA (`--grid`, `--scf-convergence`, `--keyword-line-extra`), `--out`; batch mode: point `geom` to a directory and add `--glob "*.geom" --out-dir outputs/` to render many at once.
- `mbe template`: PBS/Slurm scripts with run-control wrapper. Shared: `--scheduler [pbs|slurm]`, `--backend [qchem|orca]`, `--job-name`, `--walltime`, `--mem-gb`, `--chunk-size`, `--module`, `--command`, `--out`; PBS+qchem adds `--ncpus`, `--queue`, `--project`; Slurm+orca adds `--ncpus` (cpus-per-task), `--ntasks`, `--partition`, `--project` (account), `--qos`; `--wrapper` emits a bash submitter (bash job.sh) that writes hidden `._*.pbs/.sbatch` and submits via qsub/sbatch.
- `mbe parse <root>`: outputs → JSONL. Options: `--program [auto|qchem|orca]`, `--glob-pattern`, `--out`, `--infer-metadata`, geometry search controls (`--cluster-xyz`, `--geom-mode first|last`, `--geom-source singleton|any`, `--geom-max-lines`, `--geom-drop-ghost`, `--nosearch`). If no singleton metadata is available, it falls back to the first parsable geometry as monomer 0 for embedding.
- `mbe analyze <parsed.jsonl>`: summaries/exports. Options: `--to-csv`, `--to-xlsx`, `--plot`, `--scheme [simple|strict]`, `--max-order`.

Use `mbe <command> --help` for full flags.

## Definitions (CLI & API)

| Area | Item                         | What it does                                                                                                                                   | Key options/args                                                                                                                                                                                            | Notes                                                                                          | Implementation                                                                                                                                                                                                                                  |
| ---- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CLI  | `mbe fragment <xyz>`         | Water-heuristic fragmentation and sampling → XYZ                                                                                               | `--n`, `--seed`, `--mode random                                                                                                                                                                             | spatial`, `--require-ion`, `--prefer-special`, `--k-neighbors`, `--start-index`, `--oh-cutoff` | Spatial mode can force special fragment; writes sampled XYZ                                                                                                                                                                                     | [src/mbe_tools/cli.py](src/mbe_tools/cli.py#L12-L61)                                                                            |
| CLI  | `mbe gen <xyz>`              | Generate subset geometries up to chosen orders                                                                                                 | `--max-order` or repeatable `--order/--orders`, `--cp/--no-cp`, `--scheme`, `--backend qchem                                                                                                                | orca`, `--oh-cutoff`, `--out-dir`                                                              | Orders can be explicit list; CP toggles ghost atoms                                                                                                                                                                                             | [src/mbe_tools/cli.py](src/mbe_tools/cli.py#L64-L115)                                                                           |
| CLI  | `mbe build-input <geom>`     | Render Q-Chem/ORCA input from .geom                                                                                                            | Required `--method`, `--basis`; Q-Chem: `--thresh`, `--tole`, `--scf-convergence`, `--rem-extra`; ORCA: `--grid`, `--scf-convergence`, `--keyword-line-extra`; `--out`; batch: `--glob`, `--out-dir`        | With `--glob`, `geom` must be a directory; outputs named after stems                           | [src/mbe_tools/cli.py](src/mbe_tools/cli.py#L173-L229)                                                                                                                                                                                          |
| CLI  | `mbe template`               | Emit PBS/Slurm scripts (with run-control wrapper)                                                                                              | Shared: `--scheduler pbs                                                                                                                                                                                    | slurm`, `--backend qchem                                                                       | orca`, `--job-name`, `--walltime`, `--mem-gb`, `--chunk-size`, `--module`, `--command`, `--out`; PBS: `--ncpus`, `--queue`, `--project`; Slurm: `--ncpus`(cpus-per-task), `--ntasks`, `--partition`, `--project`(account), `--qos`; `--wrapper` | `--wrapper` writes a bash submitter that generates hidden `._*.pbs/.sbatch` then submits; run-control autodetects control files | [src/mbe_tools/cli.py](src/mbe_tools/cli.py#L101-L169) → [src/mbe_tools/hpc_templates.py](src/mbe_tools/hpc_templates.py) |
| CLI  | `mbe parse <root>`           | Parse Q-Chem/ORCA outputs to JSONL                                                                                                             | `--program auto/qchem/orca`, `--glob-pattern`, `--out`, `--infer-metadata`, `--cluster-xyz`, `--nosearch`, `--geom-mode first/last`, `--geom-source singleton/any`, `--geom-drop-ghost`, `--geom-max-lines` | Infers method/basis/grid from names/inputs; can embed cluster geometry                         | [src/mbe_tools/cli.py](src/mbe_tools/cli.py#L553-L716) → [src/mbe_tools/parsers/io.py](src/mbe_tools/parsers/io.py)                                                                                                                             |
| CLI  | `mbe analyze <parsed.jsonl>` | Summaries/exports/plots                                                                                                                        | `--to-csv`, `--to-xlsx`, `--plot`, `--scheme simple or strict`, `--max-order`                                                                                                                               | `strict` uses inclusion–exclusion; `simple` computes ΔE vs mean monomer                        | [src/mbe_tools/cli.py](src/mbe_tools/cli.py#L717-L783) → [src/mbe_tools/analysis.py](src/mbe_tools/analysis.py)                                                                                                                                 |
| CLI  | `mbe show <jsonl>`           | Quick cluster/CPU/energy view                                                                                                                  | Optional: `--monomer N` to print monomer geometry                                                                                                                                                           | Uses default JSONL selection                                                                   | [src/mbe_tools/cli.py](src/mbe_tools/cli.py#L288-L357)                                                                                                                                                                                          |
| CLI  | `mbe info <jsonl>`           | Coverage + CPU summary                                                                                                                         | Default JSONL selection                                                                                                                                                                                     | Status counts by subset_size                                                                   | [src/mbe_tools/cli.py](src/mbe_tools/cli.py#L359-L387)                                                                                                                                                                                          |
| CLI  | `mbe calc <jsonl>`           | CPU totals + MBE energies (simple/strict)                                                                                                      | `--scheme simple or strict`, `--to`, `--from`, `--monomer`, `--unit hartree/kcal/kj`                                                                                                                        | Warns on mixed program/method/basis/grid/cp combos                                             | [src/mbe_tools/cli.py](src/mbe_tools/cli.py#L389-L474)                                                                                                                                                                                          |
| CLI  | `mbe save <jsonl>`           | Archive JSONL to timestamped folder                                                                                                            | `--dest DIR`                                                                                                                                                                                                | Uses cluster_id/stamp subfolders                                                               | [src/mbe_tools/cli.py](src/mbe_tools/cli.py#L476-L505)                                                                                                                                                                                          |
| CLI  | `mbe compare <dir or glob>`  | Compare multiple JSONL runs                                                                                                                    | Optional: `--cluster ID` filter                                                                                                                                                                             | Lists cpu_ok, record counts, combo labels                                                      | [src/mbe_tools/cli.py](src/mbe_tools/cli.py#L507-L551)                                                                                                                                                                                          |
| API  | Cluster                      | `read_xyz`, `write_xyz`, `fragment_by_water_heuristic`, `fragment_by_connectivity`, `sample_fragments`, `spatial_sample_fragments`             | See function args for cutoffs, scaling, seeds                                                                                                                                                               | Supports ion retention and special-fragment preference                                         | [src/mbe_tools/cluster.py](src/mbe_tools/cluster.py)                                                                                                                                                                                            |
| API  | MBE generation               | `MBEParams`, `generate_subsets_xyz`                                                                                                            | Args: `max_order`, `orders`, `cp_correction`, `backend`, `scheme`                                                                                                                                           | Yields `(job_id, subset_indices, geom_text)` for each subset                                   | [src/mbe_tools/mbe.py](src/mbe_tools/mbe.py)                                                                                                                                                                                                    |
| API  | Input builders               | `render_qchem_input`, `render_orca_input`, `build_input_from_geom`                                                                             | Method/basis required; optional thresh/tole/scf/grid/extra lines                                                                                                                                            | Used by CLI `build-input`; accepts .geom path                                                  | [src/mbe_tools/input_builder.py](src/mbe_tools/input_builder.py)                                                                                                                                                                                |
| API  | Templates                    | `render_pbs_qchem`, `render_slurm_orca`                                                                                                        | Scheduler resources + chunking + run-control wrapper                                                                                                                                                        | `wrapper` flag mirrors CLI behavior                                                            | [src/mbe_tools/hpc_templates.py](src/mbe_tools/hpc_templates.py)                                                                                                                                                                                |
| API  | Parsing                      | `detect_program`, `parse_files`, `infer_metadata_from_path`, `glob_paths`                                                                      | Program auto-detect; metadata inference from names/inputs                                                                                                                                                   | Companion inputs help fill method/basis/grid                                                   | [src/mbe_tools/parsers/io.py](src/mbe_tools/parsers/io.py)                                                                                                                                                                                      |
| API  | Analysis                     | `read_jsonl`, `to_dataframe`, `summarize_by_order`, `compute_delta_energy`, `strict_mbe_orders`, `assemble_mbe_energy`, `order_totals_as_rows` | Convenience helpers for MBE tables and plots                                                                                                                                                                | `strict_mbe_orders` builds inclusion–exclusion rows                                            | [src/mbe_tools/analysis.py](src/mbe_tools/analysis.py)                                                                                                                                                                                          |

### CLI details with examples

| Command                  | Option(s)                                                            | Meaning                                             | Example                                                  |
| ------------------------ | -------------------------------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------- |
| `mbe fragment <xyz>`     | `--mode random/spatial`, `--n`, `--require-ion`                      | Fragment and sample XYZ                             | `mbe fragment water3.xyz --mode spatial --n 2`           |
| `mbe gen <xyz>`          | `--max-order`, `--order`, `--cp/--no-cp`                             | Generate subset geometries                          | `mbe gen big.xyz --max-order 3 --out-dir geoms`          |
| `mbe build-input <geom>` | `--backend qchem/orca`, `--method`, `--basis`                        | Render Q-Chem/ORCA input from geom                  | `mbe build-input frag.geom --backend qchem --out a.inp`  |
| `mbe template`           | `--scheduler pbs/slurm`, `--backend`, `--wrapper`                    | Emit PBS/Slurm script (optional wrapper submitter)  | `mbe template --scheduler pbs --backend qchem --wrapper` |
| `mbe parse <root>`       | `--program auto/qchem/orca`, `--glob-pattern`, geometry search flags | Parse outputs to JSONL (can embed cluster geometry) | `mbe parse ./Output --glob "*.out" --geom-source any`    |
| `mbe analyze <jsonl>`    | `--scheme simple/strict`, `--to-csv`, `--plot`                       | Summaries, exports, plots                           | `mbe analyze parsed.jsonl --scheme strict`               |
| `mbe show <jsonl>`       | `--monomer N`                                                        | Quick cluster/CPU/energy view                       | `mbe show parsed.jsonl --monomer 0`                      |
| `mbe info <jsonl>`       | (uses defaults)                                                      | Coverage + CPU summary                              | `mbe info`                                               |
| `mbe calc <jsonl>`       | `--scheme simple/strict`, `--unit hartree/kcal/kj`, `--to`, `--from` | CPU totals + MBE energies                           | `mbe calc parsed.jsonl --scheme strict --unit kcal`      |
| `mbe save <jsonl>`       | `--dest DIR`                                                         | Archive JSONL with cluster_id/timestamp             | `mbe save parsed.jsonl --dest runs/`                     |
| `mbe compare <dir        | glob>`                                                               | `--cluster ID`                                      | Compare multiple JSONL runs                              | `mbe compare runs/**/*.jsonl --cluster water20` |

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