# mbe-tools 简介

`mbe-tools` 覆盖 Many-Body Expansion (MBE) 工作流的常见环节：

- **簇与片段处理**：读取 `.xyz`，拆分片段并随机抽样（可保证包含离子）。
- **作业准备**：生成子集几何，渲染 Q-Chem / ORCA 输入文件，产出 PBS/Slurm 作业脚本（支持按批次切分提交）。
- **结果解析**：解析 ORCA/Q-Chem 输出，基于路径或伴随输入推断 method/basis/grid，写出 JSONL。
- **结果分析**：基于包含–排除的 MBE(k) 能量汇总，导出 CSV/Excel，并生成基础图表。

版本状态：**0.1.0 (MVP)**，如需调整 ghost 原子等语法，可在对应 backend 中修改。

**版本更新（P0–P5）**
- P0：全局设置加载与优先级（`mbe.toml`、环境变量），涵盖命令/模块/临时目录/队列提示。
- P1：解析自动识别程序，并从路径推断 method/basis/grid；输出检测结果与错误标签。
- P2：空间抽样模式（质心紧凑抽样，可保留离子）。
- P3：基于连接性的片段划分与标签（water/methanol/ethanol/benzene/ions）。
- P4：MBE 参数与输入构建器可调（thresh/tole/scf、grid），CLI 暴露。
- P5：PBS/Slurm 模板内置 run-control（正则确认、重试/清理、可选删除、状态 JSON）。

---

## 安装（开发模式）

```bash
cd mbe-tools
python -m pip install -e .[analysis,cli]
```

---

## 全局设置（P0）

优先级（低→高）：环境变量 → `~/.config/mbe-tools/config.toml` → `./mbe.toml` → `load_settings(path=...)` 显式指定。

支持键：`qchem_command`, `orca_command`, `qchem_module`, `orca_module`, `scratch_dir`, `scheduler_queue`, `scheduler_partition`, `scheduler_account`。

环境变量：`MBE_QCHEM_CMD`, `MBE_ORCA_CMD`, `MBE_QCHEM_MODULE`, `MBE_ORCA_MODULE`, `MBE_SCRATCH`, `MBE_SCHED_QUEUE`, `MBE_SCHED_PARTITION`, `MBE_SCHED_ACCOUNT`。

最小 `mbe.toml` 示例（按站点修改）：

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

---

## 快速上手

1) 读取与片段化 XYZ

```python
from mbe_tools.cluster import read_xyz, fragment_by_water_heuristic, fragment_by_connectivity

xyz = read_xyz("Water20.xyz")
frags = fragment_by_water_heuristic(xyz, oh_cutoff=1.25)
print(len(frags), "(water heuristic)")

# 基于连接性 + 标签（water/methanol/ethanol/benzene/ions）
frags_conn = fragment_by_connectivity(xyz, scale=1.2)
print(len(frags_conn), "(connectivity)")
```

2) 抽样片段并写出新 XYZ

```python
from mbe_tools.cluster import sample_fragments, write_xyz

picked = sample_fragments(frags, n=10, seed=42)
write_xyz("Water10_sample.xyz", picked)
```

3) 生成 MBE 子集几何

```python
from mbe_tools.mbe import MBEParams, generate_subsets_xyz

params = MBEParams(max_order=3, cp_correction=True, backend="qchem")
subset_jobs = list(generate_subsets_xyz(frags, params))
```

4) 构建输入文件（Q-Chem / ORCA）

```bash
mbe build-input water.geom --backend qchem --method wb97m-v --basis def2-ma-qzvpp --out water_qchem.inp
mbe build-input water.geom --backend orca  --method wb97m-v --basis def2-ma-qzvpp --out water_orca.inp
```

5) 生成调度脚本（PBS / Slurm）

```bash
mbe template --scheduler pbs --backend qchem --job-name mbe-qchem --chunk-size 20 --out qchem.pbs
mbe template --scheduler slurm --backend orca --job-name mbe-orca --partition work --chunk-size 10 --out orca.sbatch
```

Run-control（P5）已写入模板（可执行规范）：
- 自动查找 `<input>.mbe.control.toml`，否则 `mbe.control.toml`。
- 每次尝试先写 `job._try.out`，失败重命名为 `job.attemptN.out`，成功重命名为 `job.out`；如设置 `confirm.log_path` 则按该路径写临时日志并重命名。
- 正则确认、重试（清理/等待）、状态 `.mbe_state.json` 记录 `done/failed`、尝试次数、日志路径。
- 默认不删输出；需 `allow_delete_outputs=true` 且 `delete.enabled=true` 才会删输出，输入删除由 `delete_inputs_globs` 控制。

示例控制文件：

```toml
version = 1

[confirm]
regex_any = ['TOTAL ENERGY', 'Energy\s+=']
regex_none = ['SCF failed', 'Error']

[retry]
enabled = true
max_attempts = 2
sleep_seconds = 5
cleanup_globs = ['temp*', 'Scratch/*']
write_failed_last = true

[state]
skip_if_done = true
state_file = '.mbe_state.json'

[delete]
enabled = false
allow_delete_outputs = false
```

6) 解析输出到 JSONL

```bash
mbe parse ./Output --program auto --glob "*.out" --out parsed.jsonl
# 若目录全是同一程序，可显式 --program qchem/orca，跳过检测略快且更稳
```

7) 分析 JSONL，输出 CSV/Excel 并绘图

```bash
mbe analyze parsed.jsonl --to-csv results.csv --to-xlsx results.xlsx --plot mbe.png
```

---

## CLI 速查（含参数）

- `mbe fragment <xyz_path>`：xyz → 片段 → 抽样 → 写 xyz（当前 CLI 使用水启发式；连接性/标签请用 Python API，CLI 计划补齐）
  - `--out-xyz [sample.xyz]` 输出文件
  - `--n [10]`，`--seed [None]`
  - `--require-ion [False]` 强制包含至少一个特殊/离子片段（若存在）
  - `--mode [random|spatial]`；spatial 额外：`--prefer-special [False]`，`--k-neighbors [4]`，`--start-index [None]`
  - `--oh-cutoff [1.25]` 水启发式 O–H 阈值

- `mbe gen <xyz_path>`：生成 MBE 子集几何
  - `--out-dir [mbe_geoms]`，`--max-order [2]`，`--orders/--order`（可多次）
  - `--cp/--no-cp [cp]`，`--scheme [mbe]`，`--backend [qchem|orca]`，`--oh-cutoff [1.25]`

- `mbe build-input <geom>`：渲染 Q-Chem/ORCA 输入
  - 通用：`--backend [qchem]`，`--method`，`--basis`（必填），`--charge [0]`，`--multiplicity [1]`
  - Q-Chem：`--thresh`，`--tole`，`--scf-convergence`，`--rem-extra`
  - ORCA：`--grid`，`--scf-convergence`，`--keyword-line-extra`
  - `--out [job.inp]`

- `mbe template`：生成 PBS/Slurm 作业脚本（内置 run-control）
  - 通用：`--scheduler [pbs|slurm]`，`--backend [qchem|orca]`，`--job-name [mbe-job]`，`--walltime [24:00:00]`，`--mem-gb [32.0]`，`--chunk-size [None]`，`--module [按后端默认]`，`--command [覆写可执行]`，`--out [job.sh]`
  - PBS+qchem：`--ncpus [16]`，`--queue`，`--project`
  - Slurm+orca：`--ncpus [16]`（cpus-per-task），`--ntasks [1]`，`--partition`，`--project`(account)，`--qos`

- `mbe parse <root>`：解析输出 → JSONL
  - `--program [auto|qchem|orca]`，`--glob-pattern ["*.out"]`，`--out [parsed.jsonl]`，`--infer-metadata [True]`

- `mbe analyze <parsed.jsonl>`：汇总/导出
  - `--to-csv`，`--to-xlsx`，`--plot`
  - `--scheme [simple|strict]`，`--max-order [None]`

详细请用 `mbe <command> --help` 查看。
- `mbe analyze`：MBE(i) / ΔE 汇总，导出 csv/xlsx/plot（简单均值或严格包含–排除）

---

## 文件命名约定（子集元数据）

生成的 job_id 建议统一为 **0-based**，带阶次和哈希，避免歧义：

- 推荐：`{backend}_k{order}_f{i1}-{i2}-{i3}_{cp|nocp}_{hash}`（名字里 0-based，可零填充），如 `qchem_k2_f000-003_cp_deadbeef.out`。
- 兼容旧式：`{backend}_k{order}_{i1}.{i2}..._{hash}` 视为 1-based，再内部转 0-based。

JSON 输出的 `subset_indices` 永远是 0-based。如需人类可读的 1-based，可自行派生显示字段；解析器仍会按 0-based 处理。

## 数据模型 (JSONL)

每个计算对应一行 JSON，例如：

```json
{
  "job_id": "qchem_k2_f000-003_cp_deadbeef",
  "program": "qchem",
  "method": "wB97M-V",
  "basis": "def2-ma-QZVPP",
  "grid": "SG-2",
  "energy_hartree": -458.7018184,
  "cpu_seconds": 1234.5,
  "subset_size": 2,
  "subset_indices": [0, 3],
  "cp_correction": true,
  "raw": {"path": ".../job.out"}
}
```

---

## 示例 Notebook

`notebooks/sample_walkthrough.ipynb` 展示了：
- 从几何块构建 Q-Chem / ORCA 输入；
- 生成 PBS / Slurm 模板；
- 使用合成数据组装 MBE(k) 能量。

---

## 许可证

MIT

---

## API 速览 (Python)

- **簇与片段**（见 [src/mbe_tools/cluster.py](src/mbe_tools/cluster.py)）
  - `read_xyz(path) -> XYZ`：读取 XYZ。
  - `fragment_by_water_heuristic(xyz, oh_cutoff=1.25)`：按水分子启发式拆分。
  - `sample_fragments(fragments, n, seed=None, require_ion=False)`：随机抽样片段。
  - `write_xyz(path, fragments, comment="")`：写回 XYZ。

- **MBE 生成**（见 [src/mbe_tools/mbe.py](src/mbe_tools/mbe.py)）
  - `MBEParams(max_order, cp_correction, backend)`：配置数据类。
  - `generate_subsets_xyz(fragments, params)`：迭代产出 (job_id, subset_indices, geom_text)，支持 CP 幽灵原子。

- **输入构建**（见 [src/mbe_tools/input_builder.py](src/mbe_tools/input_builder.py)）
  - `render_qchem_input(...)` / `render_orca_input(...)`
  - `build_input_from_geom(geom_path, backend, method, basis, ...)`

- **HPC 模板**（见 [src/mbe_tools/hpc_templates.py](src/mbe_tools/hpc_templates.py)）
  - `render_pbs_qchem(job_name, walltime, ncpus, mem_gb, queue=None, project=None, module="qchem/5.2.2", input_glob="*.inp", chunk_size=None)`
  - `render_slurm_orca(job_name, walltime, ntasks, cpus_per_task, mem_gb, partition=None, account=None, qos=None, module="orca/5.0.3", command="orca", input_glob="*.inp", chunk_size=None)`

- **解析**（见 [src/mbe_tools/parsers/io.py](src/mbe_tools/parsers/io.py)）
  - `glob_paths(root, pattern)`，`parse_files(paths, program, infer_metadata=True)`
  - `infer_metadata_from_path(path)`：从路径推断 method/basis/grid。

- **MBE 计算**（见 [src/mbe_tools/mbe_math.py](src/mbe_tools/mbe_math.py)）
  - `assemble_mbe_energy(records, max_order=None)`：包含–排除 MBE(k)
  - `order_totals_as_rows(order_totals)`：转为表格行。

---

## 使用示例 (Python)

### 拆分、生成子集并构建输入

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

### 生成作业脚本

```python
from mbe_tools.hpc_templates import render_pbs_qchem, render_slurm_orca

pbs = render_pbs_qchem(job_name="mbe-qchem", chunk_size=20)
slurm = render_slurm_orca(job_name="mbe-orca", partition="work", chunk_size=10)
```

### 解析输出并计算 MBE(k)

```python
from mbe_tools.parsers.io import glob_paths, parse_files
from mbe_tools.mbe_math import assemble_mbe_energy

paths = glob_paths("./Output", "*.out")
records = parse_files(paths, program="qchem", infer_metadata=True)
mbe = assemble_mbe_energy(records, max_order=2)
print(mbe["order_totals"])  # {1: ..., 2: ...}
```
