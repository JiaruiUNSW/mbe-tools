# mbe-tools (v0.1.1)

`mbe-tools` 覆盖 Many-Body Expansion (MBE) 工作流：

- 簇与片段：读取 `.xyz`，水启发式或连接性+标签拆分，随机/空间抽样，支持保留离子。
- 作业准备：生成子集几何，渲染 Q-Chem/ORCA 输入，产出 PBS/Slurm 脚本（可分批提交，内置 run-control）。
- 解析：读取 ORCA/Q-Chem 输出，自动识别程序，基于路径或伴随输入推断 method/basis/grid，写出 JSONL。
- 分析：包含–排除 MBE(k)，汇总，CSV/Excel 导出与简单绘图。

当前状态：**v0.1.1 (MVP)**，站点相关的 ghost 原子等语法可按需在 backend 中调整。许可证：**MIT**。

---

## 安装（开发模式）

```bash
cd mbe-tools
python -m pip install -e .[analysis,cli]
```

## 全局设置优先级（P0）

优先级（低→高）：1) 环境变量 → 2) `~/.config/mbe-tools/config.toml` → 3) `./mbe.toml` → 4) `load_settings(path=...)`。

支持键：`qchem_command`, `orca_command`, `qchem_module`, `orca_module`, `scratch_dir`, `scheduler_queue`, `scheduler_partition`, `scheduler_account`。

环境变量：`MBE_QCHEM_CMD`, `MBE_ORCA_CMD`, `MBE_QCHEM_MODULE`, `MBE_ORCA_MODULE`, `MBE_SCRATCH`, `MBE_SCHED_QUEUE`, `MBE_SCHED_PARTITION`, `MBE_SCHED_ACCOUNT`。

最小 `mbe.toml` 示例：

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

## 快速上手（Python API）

1) 片段化 XYZ

```python
from mbe_tools.cluster import read_xyz, fragment_by_water_heuristic, fragment_by_connectivity

xyz = read_xyz("Water20.xyz")
frags = fragment_by_water_heuristic(xyz, oh_cutoff=1.25)
frags_conn = fragment_by_connectivity(xyz, scale=1.2)
```

2) 抽样并写回 XYZ

```python
from mbe_tools.cluster import sample_fragments, write_xyz

picked = sample_fragments(frags, n=10, seed=42)
write_xyz("Water10_sample.xyz", picked)
```

3) 生成子集几何

```python
from mbe_tools.mbe import MBEParams, generate_subsets_xyz

params = MBEParams(max_order=3, cp_correction=True, backend="qchem")
subset_jobs = list(generate_subsets_xyz(frags, params))
```

4) 构建输入

```bash
mbe build-input water.geom --backend qchem --method wb97m-v --basis def2-ma-qzvpp --out water_qchem.inp
mbe build-input water.geom --backend orca  --method wb97m-v --basis def2-ma-qzvpp --out water_orca.inp
```

5) 生成 PBS/Slurm 模板（含 run-control）

```bash
mbe template --scheduler pbs   --backend qchem --job-name mbe-qchem --chunk-size 20 --out qchem.pbs
mbe template --scheduler slurm --backend orca  --job-name mbe-orca  --partition work --chunk-size 10 --out orca.sbatch
```

6) 解析输出为 JSONL

```bash
mbe parse ./Output --program auto --glob "*.out" --out parsed.jsonl
```

7) 分析 JSONL

```bash
mbe analyze parsed.jsonl --to-csv results.csv --to-xlsx results.xlsx --plot mbe.png
```

## CLI 速查

- `mbe fragment <xyz>`：水启发式拆分+抽样 → XYZ。参数：`--out-xyz [sample.xyz]`，`--n`，`--seed`，`--require-ion`，`--mode [random|spatial]`，空间模式额外 `--prefer-special`，`--k-neighbors`，`--start-index`，`--oh-cutoff`。
- `mbe gen <xyz>`：生成子集几何。参数：`--out-dir [mbe_geoms]`，`--max-order [2]`，`--order/--orders`，`--cp/--no-cp`，`--scheme`，`--backend [qchem|orca]`，`--oh-cutoff`。
- `mbe build-input <geom>`：渲染 Q-Chem/ORCA 输入。参数：后端、必填 `--method`/`--basis`、电荷/多重度，Q-Chem (`--thresh`/`--tole`/`--scf-convergence`/`--rem-extra`)，ORCA (`--grid`/`--scf-convergence`/`--keyword-line-extra`)，`--out`。
- `mbe template`：PBS/Slurm 脚本（含 run-control）。通用：`--scheduler`，`--backend`，`--job-name`，`--walltime`，`--mem-gb`，`--chunk-size`，`--module`，`--command`，`--out`；PBS+qchem 另有 `--ncpus`，`--queue`，`--project`；Slurm+orca 另有 `--ncpus`(cpus-per-task)，`--ntasks`，`--partition`，`--project`(account)，`--qos`。
- `mbe parse <root>`：解析输出 → JSONL。参数：`--program [auto|qchem|orca]`，`--glob-pattern`，`--out`，`--infer-metadata`。
- `mbe analyze <parsed.jsonl>`：汇总/导出。参数：`--to-csv`，`--to-xlsx`，`--plot`，`--scheme [simple|strict]`，`--max-order`。

使用 `mbe <command> --help` 查看完整参数。

## Run-control（模板）

- 查找顺序：`<input>.mbe.control.toml` 优先，其次 `mbe.control.toml`，否则视为未启用。
- 尝试日志：先写 `job._try.out`，失败重命名为 `job.attemptN.out`，成功重命名为 `job.out`；`confirm.log_path` 可改临时日志路径。
- 确认：`confirm.regex_any` 必须命中且 `confirm.regex_none` 不命中，并且退出码为 0 才视为成功。
- 重试：`retry.enabled`，`max_attempts`，`sleep_seconds`，`cleanup_globs`，`write_failed_last`（将最后一次复制到 `failed_last_path`）。
- 删除保护：`delete.enabled` 且 `allow_delete_outputs=true` 才会删输出；输入仅在命中 `delete_inputs_globs` 时删除。
- 状态：`.mbe_state.json` 记录结果/次数/匹配/日志；`skip_if_done` 为真时已完成则跳过。

## 子集命名

- 推荐：`{backend}_k{order}_f{i1}-{i2}-{i3}_{cp|nocp}_{hash}`，**0-based** 片段索引（可零填充），如 `qchem_k2_f000-003_cp_deadbeef.out`。
- 兼容旧式：`{backend}_k{order}_{i1}.{i2}..._{hash}` 视为名字里的 1-based，解析时转 0-based。
JSON 输出中的 `subset_indices` 始终为 0-based。

## JSONL 模式（解析输出）

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

## API 速览

- 簇与片段（[src/mbe_tools/cluster.py](src/mbe_tools/cluster.py)）：`read_xyz`，`write_xyz`，`fragment_by_water_heuristic`，`fragment_by_connectivity`，`sample_fragments`，`spatial_sample_fragments`。
- MBE 生成（[src/mbe_tools/mbe.py](src/mbe_tools/mbe.py)）：`MBEParams`，`generate_subsets_xyz`，`qchem_molecule_block`，`orca_xyz_block`。
- 输入构建（[src/mbe_tools/input_builder.py](src/mbe_tools/input_builder.py)）：`render_qchem_input`，`render_orca_input`，`build_input_from_geom`。
- HPC 模板（[src/mbe_tools/hpc_templates.py](src/mbe_tools/hpc_templates.py)）：`render_pbs_qchem`，`render_slurm_orca`（均包含 run-control 包装）。
- 解析（[src/mbe_tools/parsers/io.py](src/mbe_tools/parsers/io.py)）：`detect_program`，`parse_files`，`infer_metadata_from_path`，`glob_paths`。
- 分析（[src/mbe_tools/analysis.py](src/mbe_tools/analysis.py)）：`read_jsonl`，`summarize_by_order`，`compute_delta_energy`，`strict_mbe_orders`。

## Notebook

`notebooks/sample_walkthrough.ipynb` 展示端到端示例：构建输入、生成模板、用合成数据组装 MBE(k)。

## 许可证

MIT
