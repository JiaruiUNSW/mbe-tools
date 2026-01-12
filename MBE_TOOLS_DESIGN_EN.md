# mbe-tools Design Document (Ideas / Architecture / API / Roadmap)

> Version: v0.1 (MVP skeleton implemented)  
> Scope: ORCA / Q-Chem Many‑Body Expansion (MBE) workflow: **Cluster design → MBE geometry/input generation (ghost atoms / CP) → Output parsing to JSON → Analysis & export (CSV/Excel/plots)**

---

## 1. Background and Goals

### 1.1 Background
Your current MBE toolkit already covers the full workflow:

- Read a large cluster (`.xyz`), split it into monomers/fragments, and sample smaller sub-clusters.
- Generate MBE subset geometries/inputs according to MBE parameters (MBE order, EE‑MBE / CP, ghost atoms, etc.).
- Extract energies, CPU time, and relevant metadata from ORCA/Q‑Chem output files.
- Perform MBE aggregation and cost/error analysis, exporting tables and plots.

### 1.2 Project Goals
`mbe-tools` should be:

1. **Installable via pip**: standard Python packaging, easy to reuse and share.
2. **A clean Python API**: stable interfaces for your scripts and notebooks.
3. **HPC-friendly CLI**: batch generate/parse/analyze with minimal friction.
4. **A unified data model**: parse ORCA and Q‑Chem into the same JSON schema.
5. **Extensible**: add parsers, ghost-atom conventions, templates, and metrics without refactoring the whole project.

### 1.3 Non-Goals
- Fully automatic, universally correct fragmentation for all chemistry (complex systems may require manual rules).
- Guaranteed stable API during 0.x (stability is expected from 1.0).
- Replacing workflow managers (Snakemake/Nextflow), though `mbe-tools` should integrate well with them.

---

## 2. Users and Typical Use Cases

### 2.1 User Profiles
- **Primary (you)**: needs reproducible analysis and a stable programmatic interface.
- **Collaborators / HPC users**: prefer CLI to handle many directories/jobs quickly.

### 2.2 Typical Use Cases
1. Read `Water20.xyz` and split into 20 water fragments (+ possible ion fragments).
2. Randomly sample `N` fragments to create `WaterN_sample.xyz` (optionally ensuring ions are included).
3. Generate MBE subset geometries for `k = 1..m` with CP ghost atoms.
4. Parse all `*.out` under `Output/` into a `parsed.jsonl` dataset.
5. Analyze `parsed.jsonl` to compute:
   - total CPU time / mean CPU time by order
   - ΔE vs reference or strict MBE(i) aggregation
   - exports to CSV/Excel and plots (ΔE vs k, cost-error, etc.)

---

## 3. High-Level Architecture (Layered Design)

### 3.1 Layer Overview

**(A) Cluster layer: geometry IO + fragmentation**  
Input: XYZ  
Output: Fragments (a unified internal representation)

**(B) MBE generation layer: subset geometry/input generation**  
Input: Fragments + MBEParams  
Output: per-subset geometry block and/or full input file (extensible)

**(C) Parsing layer: ORCA/Q‑Chem output → ParsedRecord(JSON)**  
Input: `*.out`  
Output: JSONL (one record per calculation)

**(D) Analysis layer: JSON → metrics/tables/plots**  
Input: JSONL  
Output: DataFrame, CSV, Excel, plots

### 3.2 Suggested Repository Layout
```
mbe-tools/
  pyproject.toml
  README.md
  docs/
    DESIGN.md               # this document
  src/mbe_tools/
    __init__.py
    cluster.py              # cluster design / sampling / xyz IO
    mbe.py                  # MBE parameters / subset generation
    backends/               # ghost atom conventions / input templates
      base.py
      qchem.py
      orca.py
    parsers/                # output parsers
      base.py
      qchem.py
      orca.py
      io.py
    analysis.py             # JSON -> aggregation/metrics/export
    cli.py                  # CLI entry point
    templates/              # optional: pbs/slurm/qchem/orca templates
  tests/
```

---

## 4. Unified Data Model

### 4.1 Geometry Layer Types
- `Atom(element, x, y, z)`
- `Fragment = List[Atom]`
- `XYZ(comment, atoms)`

### 4.2 ParsedRecord (Parsing Output Schema)
Each calculation should become a single JSON object (stored as JSON Lines):

```json
{
  "job_id": "orca_k2_ab12cd34",
  "program": "orca",
  "path": "/.../job.out",

  "energy_hartree": -458.7018184,
  "cpu_seconds": 1234.5,

  "method": "wB97M-V",
  "basis": "def2-ma-QZVPP",
  "grid": "DEFGRID3",

  "subset_size": 2,
  "subset_indices": [0, 3],
  "cp_correction": true,

  "extra": {
    "wall_seconds": 1300.1,
    "host": "nid002024",
    "notes": "..."
  }
}
```

Notes:
- In v0.1, some metadata (method/basis/grid/subset indices) may be missing depending on your output and naming conventions.
- The schema keeps fields optional but reserved so you can fill them later via path parsing, manifest files, or input-file reading.

---

## 5. Modules and Responsibilities

### 5.1 `mbe_tools.cluster`
**Responsibility**: XYZ IO, fragmentation, random sampling.

**Public API (recommended stable)**
- `read_xyz(path) -> XYZ`
- `write_xyz(path, fragments, comment="")`
- `fragment_by_water_heuristic(xyz, oh_cutoff=1.25) -> List[Fragment]`
- `sample_fragments(fragments, n, seed=None, require_ion=False) -> List[Fragment]`

**Planned extensions**
- `fragment_by_connectivity(...)` for general molecules
- fragment labels (`water/anion/cation/organic`)
- `preserve_original_order=True` option

---

### 5.2 `mbe_tools.mbe` + `mbe_tools.backends`
**Responsibility**: Generate subset geometries/inputs (ghost atoms / CP).

**Public API**
- `MBEParams(max_order, cp_correction, backend, charge, multiplicity)`
- `generate_subsets_xyz(fragments, params, orders=None) -> Iterable[(job_id, subset_indices, geom_block)]`
- (optional helpers) `qchem_molecule_block(...)`, `orca_xyz_block(...)`

**Backends (ghost atom formatting)**
- `QChemBackend.format_atom(atom, ghost=False) -> str`
  - MVP: ghost atoms are represented as `@H`, `@O`, ...
- `OrcaBackend.format_atom(atom, ghost=False) -> str`
  - MVP: ghost atoms represented using a common pattern like `H : x y z` (must be aligned to your ORCA input style)

**Planned extensions**
- Generate full `.inp/.in` files (keywords + template)
- Different ghost strategies for EE‑MBE vs CP‑MBE
- Template system (Jinja2 or pure formatting)

---

### 5.3 `mbe_tools.parsers`
**Responsibility**: Extract analysis-ready information from output files.

**Public API**
- `parse_files(paths, program) -> List[ParsedRecord]`
- `glob_paths(root, pattern) -> List[str]`

**Parsing Strategy**
- Use robust output markers (FINAL ENERGY, total runtime) via regex first.
- For method/basis/grid/subset metadata:
  1) parse from the corresponding input file (`.inp/.in`) when available; or
  2) infer from file/folder naming conventions; or
  3) use a sidecar manifest (`mapping.json`: `job_id -> params`).

**Planned extensions**
- Support more engines (Psi4, Gaussian, etc.)
- Walltime extraction from Slurm/PBS (sacct/qstat)
- Failure classification (SCF not converged, etc.)

---

### 5.4 `mbe_tools.analysis`
**Responsibility**: JSONL → aggregated metrics → exports and plots.

**MVP Public API**
- `read_jsonl(path) -> List[dict]`
- `to_dataframe(records) -> pd.DataFrame`
- `summarize_by_order(df) -> df_summary`
- `compute_delta_energy(df, reference_order=1) -> df`

**Important: strict MBE aggregation (future work)**
The MVP delta energy is a convenience metric. Proper MBE(i) usually requires inclusion–exclusion aggregation:
- Build `MBE(1), MBE(2), ...` from subset energies.
- Compute increments `ΔMBE(i) = MBE(i) - MBE(i-1)`.
- Support CP/EE-MBE rules consistently.

Recommended future module:
- `mbe_tools/mbe_math.py`
  - `assemble_mbe_energy(records, max_order, scheme="inclusion-exclusion")`
  - `compute_increment(records, i)`
  - `compute_cost_metrics(records)`

---

## 6. CLI Design (HPC-Oriented)

### 6.1 Implemented Commands (MVP)
- `mbe fragment big.xyz --n 10 --out-xyz sample.xyz [--require-ion]`
- `mbe gen big.xyz --max-order 3 --backend qchem --out-dir mbe_geoms [--cp/--no-cp]`
  - outputs `.geom` coordinate blocks per subset
- `mbe parse ./Output --program qchem --glob "*.out" --out parsed.jsonl` (supports `--program auto`; emits `program_detected`, `status`, `wall_seconds/cpu_seconds`, `error_reason` when parsing fails)
- `mbe analyze parsed.jsonl --to-csv results.csv --to-xlsx results.xlsx --plot mbe.png`

### 6.2 Proposed CLI Extensions
- `mbe template pbs|slurm ...` (job script generation)
- `mbe build-input qchem|orca ...` (full input generation)
- `mbe assemble parsed.jsonl --scheme inclusion-exclusion --max-order 5` (strict aggregation)
- `mbe validate parsed.jsonl` (missing fields / failed jobs / outliers)

---

## 7. Roadmap and Feature Backlog

### 7.1 v0.1 (Delivered MVP)
- [x] Standard package structure + `pyproject.toml`
- [x] XYZ IO + water heuristic fragmentation + random sampling
- [x] Subset geometry generation (with ghost atoms)
- [x] ORCA/Q‑Chem energy + CPU time parsing → JSONL
- [x] JSONL → pandas summary + simple ΔE + CSV/Excel/plot export
- [x] CLI: fragment / gen / parse / analyze

### 7.2 v0.2 (Recommended next)
- [x] **Metadata inference from naming conventions**: fill `subset_indices`, `subset_size`, `cp_correction`, `method`, `basis` when tokens exist
- [x] **Strict MBE(i)/ΔMBE(i)**: inclusion–exclusion aggregation + CLI `--scheme strict`
- [ ] **Full input generation**: Q‑Chem `.inp` + ORCA `.in` via templates + `mbe build-input`
- [ ] Better unit tests with small sample outputs (reference geoms/outputs)

### 7.3 v0.3 (Quality-of-life and research-grade features)
- [ ] Slurm/PBS walltime parsing (sacct/qstat)
- [ ] Failure reason classification (SCF/opt/IO)
- [ ] cost-error plots across methods/bases/grids
- [ ] Plugin architecture for parsers/backends (entry points)

---

## 8. Quality, Conventions, and Versioning

### 8.1 Code Standards
- Python >= 3.10
- Recommended: `ruff` + `black` later
- Type hints on stable interfaces

### 8.2 Test Strategy
- Parser tests with minimal output samples (even truncated)
- Math tests for strict MBE aggregation (ΔMBE, inclusion–exclusion)
- CLI smoke tests

### 8.3 Versioning
- 0.x: API may evolve
- 1.0: stabilize the core API (`cluster`, `mbe`, `parsers`, `analysis`)

---

## 9. Implementation Notes (Recommended Next Steps)

### 9.1 Prioritize “naming convention → metadata” (Done for MVP)
Implemented `infer_metadata_from_path(path)` in `parsers/io.py` to populate subset/order/CP/method/basis when tokens exist. Next: optionally read companion input files (`.inp/.in`) to backfill missing metadata.

### 9.1b Global settings (P0)
`mbe_tools/config.py` provides `Settings`, `load_settings()`, `get_settings()`, and `use_settings()` context manager. Precedence: environment → `~/.config/mbe-tools/config.toml` → `./mbe.toml` → explicit path. Keys cover commands/modules/scratch and scheduler defaults.

### 9.2 Add “full input generation” (Next)
Add templates under `mbe_tools/templates/` (Q-Chem/ORCA) and a CLI command `mbe build-input` to emit ready inputs from geometry blocks.

### 9.3 Strict MBE aggregation (Done)
`mbe_math.py` implements inclusion–exclusion aggregation; CLI supports `--scheme strict` to print MBE(k) table. Future: cost/error metrics and missing-subset reporting in summaries.

### 9.4 Spatial sampling (P2)
Added `FragmentRecord` with centroid/special metadata and `spatial_sample_fragments` for greedy nearest-neighbor selection. CLI `mbe fragment` now supports `--mode spatial --prefer-special --k-neighbors 4 --start-index N` to generate compact subsets that include ions/special fragments when requested.

### 9.5 Connectivity fragmentation and labels (P3)
Added covalent-radii connectivity (`fragment_by_connectivity`) with basic pattern labeling (water/methanol/ethanol/benzene) and ion marking. Tests cover multi-molecule splits and benzene ring connectivity.

### 9.6 MBE parameterization and inputs (P4)
Extended `MBEParams` with `orders`, `scheme`, and common SCF thresholds; `generate_subsets_xyz` honors explicit orders. Input builders accept Q-Chem `thresh`/`tole`/`scf_convergence` and ORCA `grid`/`scf_convergence`, surfaced via CLI `mbe gen`/`mbe build-input` options.

---

## 10. Appendix: Current MVP Status
The v0.1 MVP provides a working skeleton and CLI. Backend ghost-atom syntax and strict MBE aggregation are intentionally left as “configurable / extendable” parts for v0.2+.
