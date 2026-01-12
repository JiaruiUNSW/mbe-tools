
# mbe-tools

`mbe-tools` is a Python package that covers the full **Many-Body Expansion (MBE)** loop:

- **Cluster design**: read `.xyz`, extract fragments, and sample subsets (with optional ion retention).
- **MBE job prep**: generate subset geometries, build Q-Chem/ORCA inputs, and emit PBS/Slurm job scripts (with optional chunked submission).
- **Parsing**: read ORCA / Q-Chem outputs, infer method/basis/grid metadata from paths or companion inputs, and write JSONL.
- **Analysis**: inclusion–exclusion MBE(k), summaries, CSV/Excel export, and basic plots.

Status: **0.1.0 (MVP)**. Backend syntax (e.g., ghost atoms) may need local tweaks.

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

## Quickstart

### 1) Parse an XYZ and fragment it

```python
from mbe_tools.cluster import read_xyz, fragment_by_water_heuristic, fragment_by_connectivity

xyz = read_xyz("Water20.xyz")
frags = fragment_by_water_heuristic(xyz, oh_cutoff=1.25)
print(len(frags), "fragments (water heuristic)")

# Connectivity + labeling (water/methanol/ethanol/benzene/ions)
frags_conn = fragment_by_connectivity(xyz, scale=1.2)
print(len(frags_conn), "fragments (connectivity)")
```

### 2) Randomly sample N fragments and write a new XYZ

```python
from mbe_tools.cluster import sample_fragments, write_xyz

picked = sample_fragments(frags, n=10, seed=42)
write_xyz("Water10_sample.xyz", picked)
```

### 3) Generate MBE geometries (subsets)

```python
from mbe_tools.mbe import MBEParams, generate_subsets_xyz

params = MBEParams(max_order=3, cp_correction=True, backend="qchem")
subset_jobs = list(generate_subsets_xyz(frags, params))
# subset_jobs yields (job_id, subset_frag_indices, xyz_text)
```

### 4) Build inputs (Q-Chem/ORCA)

Prepare a geometry block file (e.g., `water.geom`), then:

```bash
mbe build-input water.geom --backend qchem --method wb97m-v --basis def2-ma-qzvpp --out water_qchem.inp
mbe build-input water.geom --backend orca  --method wb97m-v --basis def2-ma-qzvpp --out water_orca.inp
```

### 5) Emit scheduler templates (PBS/Slurm)

```bash
mbe template --scheduler pbs --backend qchem --job-name mbe-qchem --chunk-size 20 --out qchem.pbs
mbe template --scheduler slurm --backend orca --job-name mbe-orca --partition work --chunk-size 10 --out orca.sbatch
```

Run-control (P5) is baked into the emitted templates (行为概要/可执行规范):
- Detects control files named `<input>.mbe.control.toml` or `mbe.control.toml`.
- Temporary log per attempt: writes `job._try.out`; on failure renames to `job.attemptN.out`; on success renames to `job.out`.
- Supports regex confirmation on the temporary log (default path above unless `confirm.log_path` overrides), retries with cleanup/sleep, and state tracking in `.mbe_state.json`.
- Can optionally delete inputs/outputs on success (guarded by `allow_delete_outputs`).

Example control file:

```toml
version = 1

[confirm]
regex_any = ['TOTAL ENERGY', 'Energy\s+=']
regex_none = ['SCF failed', 'Error']

[retry]
enabled = true
max_attempts = 2
sleep_seconds = 5
cleanup_globs = ["temp*", "Scratch/*"]
write_failed_last = true

[state]
skip_if_done = true

[delete]
enabled = false
allow_delete_outputs = false
```

### 6) Parse outputs to JSONL

```bash
mbe parse ./Output --program auto --glob "*.out" --out parsed.jsonl
# 若目录全是同一程序，可显式指定 --program qchem 或 --program orca 以跳过检测、略微提速
```

### 7) Analyze JSONL to CSV/Excel + plot

```bash
mbe analyze parsed.jsonl --to-csv results.csv --to-xlsx results.xlsx --plot mbe.png
```

---

-## CLI

After installing with `[cli]`, you get a `mbe` command. Key commands and parameters (defaults in `[]`):

- `mbe fragment <xyz_path>` → fragment + sample → write XYZ（当前 CLI 使用水启发式；连接性/标签请参考 Quickstart 的 Python 示例，后续 CLI 将补齐）
  - `--out-xyz [sample.xyz]` output path
  - `--n [10]`, `--seed [None]`
  - `--require-ion [False]` ensure at least one special/ion if present
  - `--mode [random|spatial]`; spatial extras: `--prefer-special [False]`, `--k-neighbors [4]`, `--start-index [None]`
  - `--oh-cutoff [1.25]` water heuristic O–H cutoff

- `mbe gen <xyz_path>` → MBE subset geometries
  - `--out-dir [mbe_geoms]`, `--max-order [2]`, `--orders/--order` (repeatable), `--cp/--no-cp [cp]`, `--scheme [mbe]`, `--backend [qchem|orca]`, `--oh-cutoff [1.25]`

- `mbe build-input <geom>` → render Q-Chem/ORCA input
  - Common: `--backend [qchem]`, `--method (required)`, `--basis (required)`, `--charge [0]`, `--multiplicity [1]`
  - Q-Chem: `--thresh`, `--tole`, `--scf-convergence`, `--rem-extra`
  - ORCA: `--grid`, `--scf-convergence`, `--keyword-line-extra`
  - `--out [job.inp]`

- `mbe template` → PBS/Slurm scripts with run-control wrapper
  - Common: `--scheduler [pbs|slurm]`, `--backend [qchem|orca]`, `--job-name [mbe-job]`, `--walltime [24:00:00]`, `--mem-gb [32.0]`, `--chunk-size [None]`, `--module [auto per backend]`, `--command [override binary]`, `--out [job.sh]`
  - PBS+qchem: `--ncpus [16]`, `--queue`, `--project`
  - Slurm+orca: `--ncpus [16]` (cpus-per-task), `--ntasks [1]`, `--partition`, `--project` (account), `--qos`

- `mbe parse <root>` → outputs → JSONL
  - `--program [auto|qchem|orca]`, `--glob-pattern ["*.out"]`, `--out [parsed.jsonl]`, `--infer-metadata [True]`

- `mbe analyze <parsed.jsonl>` → summaries/exports
  - `--to-csv`, `--to-xlsx`, `--plot`
  - `--scheme [simple|strict]`, `--max-order [None]`

Run `mbe <command> --help` for complete details.
- `mbe analyze` – compute MBE(i), ΔE, CPU time summaries → csv/xlsx/plot (simple mean or strict inclusion–exclusion)

Run help:

```bash
mbe --help
mbe fragment --help
```

---

## Filename convention for subset metadata

Generated job IDs now include order and **0-based** fragment indices plus a hash for stability. Use one canonical pattern to avoid ambiguity:

- Recommended: `{backend}_k{order}_f{i1}-{i2}-{i3}_{cp|nocp}_{hash}` with 0-based indices in the name (zero-padding allowed), e.g. `qchem_k2_f000-003_cp_deadbeef.out`.
- Legacy (still parsed): `{backend}_k{order}_{i1}.{i2}..._{hash}` treated as 1-based indices in the string but converted to 0-based internally.

JSON outputs always expose `subset_indices` as 0-based. If you keep 1-based names for human readability, be aware the parser will convert them to 0-based fields.

## Data model (JSONL)

Each parsed calculation becomes one JSON object per line. Key fields surfaced by `mbe parse`:

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

- `program_detected` is auto-set by the parser implementation; `program` is the requested parser backend.
- `status` is set to `"ok"` when energy is parsed successfully; `error_reason` carries a short tag if parsing fails.
- `wall_seconds` is parsed for ORCA runtime blocks and falls back to `cpu_seconds` when only CPU time is available.

---

## Extending / customizing

- Add or override parsers in `mbe_tools/parsers/*`
- Adjust ghost-atom formatting in `mbe_tools/backends/*`
- Add new metrics in `mbe_tools/analysis.py`

---

## Sample notebook

A lightweight walkthrough lives at `notebooks/sample_walkthrough.ipynb` showing how to:
- build Q-Chem/ORCA inputs from a geometry block,
- generate PBS/Slurm templates,
- assemble MBE(k) energies from synthetic data.

---

## License

MIT

---

## API reference (Python) — functions与缺省行为

**Cluster helpers** ([src/mbe_tools/cluster.py](src/mbe_tools/cluster.py))
- `read_xyz(path)`: 严格读取 XYZ，坏行/数量直接报错。
- `write_xyz(path, fragments, comment="Generated by mbe-tools")`: 扁平化所有片段写出，默认注释。
- `fragment_by_water_heuristic(xyz, oh_cutoff=1.25)`: 每个 O 选最近的两颗 H（<=cutoff）成水分子；剩余原子各自成单原子片段并标记 `special=True`。
- `fragment_by_connectivity(xyz, scale=1.2)`: 共价半径×scale 建立键，连通分量即片段；自动标签 water/methanol/ethanol/benzene/ions，否则 unknown。
- `sample_fragments(fragments, n, seed=None, require_ion=False)`: 随机采样 `n`；当 `require_ion=True` 且存在 special 片段时强制包含至少一个。
- `spatial_sample_fragments(fragments, n, seed=None, prefer_special=True, k_neighbors=4, start="random", start_index=None)`: 基于质心的贪心紧凑采样；默认偏向 special，起点随机（或 `start="special"` 用第一个特殊片段）。

**MBE generation** ([src/mbe_tools/mbe.py](src/mbe_tools/mbe.py))
- `MBEParams(max_order=2, orders=None, cp_correction=True, backend="qchem", charge=0, multiplicity=1, scheme="mbe", thresh=None, tole=None, scf_convergence=None)`：参数容器。
- `generate_subsets_xyz(fragments, params, orders=None)`: 目标阶次默认 `params.orders`，缺省为 `1..max_order`；非子集原子在 `cp_correction=True` 时转 ghost；
  - job_id 模式 `{backend}_k{order}_{1-based-indices}_{md5}`（md5 前缀可复现）。
- `qchem_molecule_block(geom_block, charge=0, multiplicity=1)` / `orca_xyz_block(geom_block)`：包裹坐标块。

**Input builders** ([src/mbe_tools/input_builder.py](src/mbe_tools/input_builder.py))
- `render_qchem_input(geom_block, method, basis, charge=0, multiplicity=1, thresh=None, tole=None, scf_convergence=None, rem_extra=None)`: 可选项为 `None` 时不写出，`rem_extra` 行按原样附在 `$rem`。
- `render_orca_input(geom_block, method, basis, charge=0, multiplicity=1, grid=None, scf_convergence=None, keyword_line_extra=None)`: 头行 `!` 追加可选 token（grid/SCF/附加关键字）。
- `build_input_from_geom(geom_path, backend, ...)`: 读取坐标并分派到 Q-Chem/ORCA 渲染；未知 backend 抛错。

**HPC templates & run-control** ([src/mbe_tools/hpc_templates.py](src/mbe_tools/hpc_templates.py))
- `render_pbs_qchem(job_name, walltime="24:00:00", ncpus=16, mem_gb=32.0, queue=None, project=None, module="qchem/5.2.2", input_glob="*.inp", chunk_size=None)`: chunk 模式每批写子 PBS 并 `qsub`；默认加载模块、设置 `NCPUS/MEM`。
- `render_slurm_orca(job_name, walltime="24:00:00", ntasks=1, cpus_per_task=16, mem_gb=32.0, partition=None, account=None, qos=None, module="orca/5.0.3", command="orca", input_glob="*.inp", chunk_size=None)`: 类似逻辑，默认 `OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}`。
- 两者都会生成 `run_with_control`：优先 `<input>.mbe.control.toml`，否则 `mbe.control.toml`；`WRAPPER_LOG` 默认 `.mbe_wrapper.log`；命令默认 `QC_CMD=qchem`、`ORCA_CMD=orca`。
- Run-control 缺省（无控制文件）：不做正则确认，重试/删除关闭，状态文件 `.mbe_state.json` 默认跳过，严格模式关闭。存在控制文件时：
  - 每次尝试先写 `job._try.out`，失败重命名为 `job.attemptN.out`，成功重命名为 `job.out`；可用 `confirm.log_path` 改写临时日志路径。
  - `confirm.regex_any/regex_none` 决定成功判断。
  - `retry.enabled/max_attempts/sleep_seconds/cleanup_globs/write_failed_last` 控制重试与清理。
  - `delete.enabled` + `allow_delete_outputs` 才会删输出；输入删除受 `delete_inputs_globs`。
  - `state.skip_if_done` 默认 true，会在 `.mbe_state.json` 记录 `done/failed`、尝试次数、日志路径。
  - 解析失败且 `template.strict=true` 时直接退出，否则回落到默认配置并记录警告。

**Parsing** ([src/mbe_tools/parsers/io.py](src/mbe_tools/parsers/io.py))
- `detect_program(text)`: 返回 `qchem/orca/unknown`。
- `glob_paths(root, pattern)`: 排序后的 glob 结果。
- `infer_metadata_from_path(path)`: 从文件名推断 `subset_size` (`k` token)、`subset_indices`（推荐 f-token 0-based；旧式点分视为 1-based 再转 0-based）、`cp_correction`，以及常见 method/basis/grid（可缺省）。
- `parse_files(paths, program, infer_metadata=True)`: `program="auto"` 时按内容检测；指定 `qchem/orca` 时跳过检测更快更稳。默认 `infer_metadata=True` 会套用路径推断，并在需要时尝试配套输入文件 `.inp/.in` 填充 method/basis/grid；设置 `program_detected/status/error_reason`。

**Analysis helpers** ([src/mbe_tools/analysis.py](src/mbe_tools/analysis.py))
- `read_jsonl(path)`: 读取非空行 JSON。
- `to_dataframe(records)`: 需要 pandas（`mbe-tools[analysis]`）。
- `summarize_by_order(df)`: 按 `subset_size` 汇总 count/min/max/CPU 总和与均值。
- `compute_delta_energy(df, reference_order=1)`: 相对参考阶次的能量差列。
- `strict_mbe_orders(records, max_order=None)`: 用 `assemble_mbe_energy` 做包含–排除，返回行列表和缺失子集列表。

---

## Examples (Python)

### Fragment, generate subsets, build inputs

```python
from mbe_tools.cluster import read_xyz, fragment_by_water_heuristic, sample_fragments
from mbe_tools.mbe import MBEParams, generate_subsets_xyz
from mbe_tools.input_builder import render_qchem_input

xyz = read_xyz("Water20.xyz")
frags = fragment_by_water_heuristic(xyz)
picked = sample_fragments(frags, n=10, seed=7)
params = MBEParams(max_order=2, cp_correction=True, backend="qchem")

job_id, subset, geom = next(generate_subsets_xyz(picked, params))
inp_text = render_qchem_input(geom, method="wb97m-v", basis="def2-ma-qzvpp")
with open(f"{job_id}.inp", "w", encoding="utf-8") as f:
    f.write(inp_text)
```

### Generate job scripts

```python
from mbe_tools.hpc_templates import render_pbs_qchem, render_slurm_orca

pbs = render_pbs_qchem(job_name="mbe-qchem", chunk_size=20)
slurm = render_slurm_orca(job_name="mbe-orca", partition="work", chunk_size=10)
```

### Parse outputs and compute MBE(k)

```python
from mbe_tools.parsers.io import glob_paths, parse_files
from mbe_tools.mbe_math import assemble_mbe_energy

paths = glob_paths("./Output", "*.out")
records = parse_files(paths, program="qchem", infer_metadata=True)
mbe = assemble_mbe_energy(records, max_order=2)
print(mbe["order_totals"])  # {1: ..., 2: ...}
```

