# Test Plan for mbe-tools

This plan enumerates the requested test cases, suggested fixtures, and file organization. Tests assume `pytest`.

## Directory Layout (suggested)
- `tests/fixtures/xyz/` – small XYZ inputs (water clusters, methanol/ethanol, benzene dimers, ions).
- `tests/fixtures/control/` – run-control TOML examples (valid, invalid, strict/non-strict).
- `tests/fixtures/outputs/qchem/` – minimal Q-Chem output snippets for parse tests.
- `tests/fixtures/outputs/orca/` – minimal ORCA output snippets for parse tests.
- `tests/fixtures/jsonl/` – sample parsed records for analysis/export tests.

## Test Cases
| Test Name                                                 | Test Scope                                       | Expected Results                                                                    |
| --------------------------------------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------- |
| TC_PKG_001_ImportLibrary                                  | Import mbe_tools and basic modules               | `import mbe_tools` succeeds; `__version__` exists; no syntax/import errors          |
| TC_P0_001_Settings_Defaults                               | Load settings with no files/env                  | Settings load with documented defaults; no crash                                    |
| TC_P0_002_Settings_Precedence_EnvOverrides                | Env vars override global defaults                | Env values (e.g., `MBE_QCHEM_CMD`) override config defaults                         |
| TC_P0_003_Settings_Precedence_ProjectOverridesUser        | Project `./mbe.toml` overrides user config       | Project config wins over `~/.config/...`                                            |
| TC_P0_004_Settings_Propagation_Backends                   | Settings used by render/template functions       | Generated scripts/inputs reference configured ORCA/QChem command/module paths       |
| TC_XYZ_001_ReadWrite_RoundTrip                            | `read_xyz()` then `write_xyz()` then re-read     | Atom count preserved; coordinates preserved within tolerance; comment preserved     |
| TC_XYZ_002_ReadXYZ_InvalidAtomCount                       | XYZ header count mismatches body lines           | Raises clear error (no silent corruption)                                           |
| TC_CLUSTER_001_Fragment_WaterHeuristic_Basic              | `fragment_by_water_heuristic()` on water cluster | Returns N fragments; each water fragment is OHH; no lost atoms                      |
| TC_CLUSTER_002_Fragment_WaterHeuristic_IonicWater         | Heuristic with Na+/Cl- present                   | Ions become single-atom fragments; water fragments remain OHH; ions flagged special |
| TC_CLUSTER_003_Fragment_Connectivity_Methanol             | Connectivity fragmentation on methanol           | Each methanol molecule is one fragment; no radical splits                           |
| TC_CLUSTER_004_Fragment_Connectivity_Ethanol              | Connectivity fragmentation on ethanol            | Each ethanol is one fragment                                                        |
| TC_CLUSTER_005_Fragment_Connectivity_Benzene              | Connectivity fragmentation on benzene dimer      | Each benzene is one fragment; ring intact                                           |
| TC_CLUSTER_006_LabelDetection                             | Labeling for water/methanol/ethanol/benzene/ion  | Expected labels assigned                                                            |
| TC_SAMPLE_001_Sample_Random_Reproducible                  | Random sampling with seed                        | Same seed → identical fragment order/indices                                        |
| TC_SAMPLE_002_Sample_Random_RequireIon                    | Random sampling with `require_ion=True`          | If ions exist: selection includes ≥1 ion; else normal sampling                      |
| TC_SAMPLE_003_Sample_Spatial_Reproducible                 | Spatial sampling with seed                       | Same seed → same start and subset; spatially compact                                |
| TC_SAMPLE_004_Sample_Spatial_PreferSpecial                | Spatial sampling with `prefer_special=True`      | Special fragments included when present                                             |
| TC_MBE_001_Subsets_Count_MaxOrder                         | `generate_subsets_xyz(..., max_order=K)`         | Correct subset counts per order; no duplicates                                      |
| TC_MBE_002_Subsets_OrdersArray                            | `orders=[1,3]` only                              | Only order-1 and order-3 subsets generated                                          |
| TC_MBE_003_JobId_Format_FToken0Based                      | job_id uses `backend_k{order}_f..._{cp           | nocp}_{hash}`                                                                       | Pattern matches recommended 0-based f-token format |
| TC_MBE_004_CP_GhostAtoms_QChem                            | CP on + Q-Chem formatting                        | Non-subset atoms as ghosts with `@Element`; subset atoms normal                     |
| TC_MBE_005_CP_GhostAtoms_ORCA                             | CP on + ORCA formatting                          | Non-subset atoms as ghosts using `Element :` syntax                                 |
| TC_MBE_006_NoCP_NoGhosts                                  | CP off                                           | Geometry contains only subset atoms                                                 |
| TC_RENDER_001_RenderInputs_QChem_ThreshTolE               | Render Q-Chem input with thresh/tole             | Input contains correct THRESH/SCF_CONVERGENCE/TolE fields                           |
| TC_RENDER_002_RenderInputs_ORCA_ThreshTolE                | Render ORCA input with thresh/tole               | Input contains correct ORCA keywords/blocks                                         |
| TC_P5_001_Control_Discovery_PerInputOverridesDir          | `<input>.mbe.control.toml` vs `mbe.control.toml` | Per-input control wins over directory default                                       |
| TC_P5_002_Control_Missing_DisablesRunControl              | No control file present                          | Wrapper runs once; no confirm/retry/delete/state                                    |
| TC_P5_003_Control_Invalid_StrictFalse                     | Invalid TOML, strict=false                       | Wrapper warns, disables run-control, job still runs                                 |
| TC_P5_004_Control_Invalid_StrictTrue                      | Invalid TOML, strict=true                        | Wrapper exits non-zero; no deletion                                                 |
| TC_P5_005_Logging_Rename_FailKeepsAttempt_SuccessIsJobOut | Attempt logging rule                             | Attempt writes `job._try.out`; fail → `job.attemptN.out`; success → `job.out`       |
| TC_P5_006_Confirm_Success_MatchesRegexAny                 | confirm success path                             | Success marker stops retries                                                        |
| TC_P5_007_Confirm_Fail_RegexNoneOverrides                 | confirm failure via regex_none                   | Failure marker forces failure even if success marker present                        |
| TC_P5_008_Retry_BoundedAttempts                           | Retry loop bounds                                | Attempts ≤ 1 + max_attempts; attempt logs preserved                                 |
| TC_P5_009_StateFile_Written                               | `.mbe_state.json` written                        | Status, attempts, confirmed, final_log, attempt_logs, timestamp recorded            |
| TC_P5_010_SkipIfDone                                      | `skip_if_done=true`                              | When state marks done, wrapper skips and exits 0                                    |
| TC_P5_011_Delete_DefaultInputsOnly                        | Delete safety defaults                           | Only inputs deleted on success; outputs untouched by default                        |
| TC_P5_012_Delete_OutputsRequireAllowGate                  | Output deletion gates                            | Outputs deleted only when `allow_delete_outputs=true` and globs provided            |
| TC_TEMPLATE_001_RenderPBS_QChem_ChunkSize                 | `render_pbs_qchem(chunk_size=N)`                 | Inputs split into chunks of N; directives present                                   |
| TC_TEMPLATE_002_RenderSlurm_ORCA_ChunkSize                | `render_slurm_orca(chunk_size=N)`                | Chunking correct; Slurm directives present                                          |
| TC_PARSE_001_AutoDetect_QChem                             | Auto-detect on Q-Chem output                     | `program_detected="qchem"`; energy parsed                                           |
| TC_PARSE_002_AutoDetect_ORCA                              | Auto-detect on ORCA output                       | `program_detected="orca"`; energy parsed                                            |
| TC_PARSE_003_Parse_QChem_EnergyCpu                        | Parse Q-Chem fields                              | `energy_hartree` correct; `cpu_seconds` parsed/None with reason; status correct     |
| TC_PARSE_004_Parse_ORCA_EnergyCpu                         | Parse ORCA fields                                | `energy_hartree` and `cpu_seconds` correct; status correct                          |
| TC_PARSE_005_Metadata_FromFilename_SubsetIndices          | Infer from f000-003 token                        | JSON has `subset_indices: [0,3]`, `subset_size=2`, cp flag consistent               |
| TC_PARSE_006_JSONSchema_RequiredFields                    | Validate JSONL schema                            | Required keys present; types correct                                                |
| TC_ANALYSIS_001_ReadJsonl_Load                            | `read_jsonl()` loads                             | Returns list of dicts; ignores blank lines safely                                   |
| TC_ANALYSIS_002_GroupByOrder_Aggregates                   | Group-by order summary                           | Correct counts and CPU totals/means                                                 |
| TC_ANALYSIS_003_DeltaEnergy_Reference                     | `compute_delta_energy`                           | Adds delta col; reference order delta ≈ 0; others as defined                        |
| TC_EXPORT_001_ExportCSV                                   | Export to CSV                                    | CSV exists; columns correct; row count matches                                      |
| TC_EXPORT_002_ExportExcel                                 | Export to XLSX                                   | XLSX exists; numeric columns typed                                                  |
| TC_EXPORT_003_PlotGeneration                              | Plot export                                      | Image produced; axes labels present; no display required                            |
| TC_CLI_001_CLI_Fragment_WaterHeuristic                    | `mbe fragment ...` (heuristic)                   | Writes XYZ; fragment count matches; deterministic with seed                         |
| TC_CLI_002_CLI_Gen_Subsets                                | `mbe gen ... --backend qchem/orca`               | Outputs subset geometries; count and job_id naming correct                          |
| TC_CLI_003_CLI_Parse_Auto                                 | `mbe parse ... --program auto`                   | JSONL produced; mixed outputs handled; `program_detected` present                   |
| TC_CLI_004_CLI_Analyze                                    | `mbe analyze ...` pipeline                       | Summary/exports created; delta energy computed when requested                       |

## Notes on Fixtures
- **XYZ**: include small water clusters, methanol/ethanol monomers, benzene dimer, and an ion (Na/Cl) case.
- **Control TOML**: valid confirm+retry, invalid syntax, strict=true/false, delete outputs gated vs default.
- **Outputs**: trimmed Q-Chem/ORCA outputs containing energy, CPU/wall time markers, success/failure phrases for regex tests.
- **JSONL**: small parsed set for analysis/export tests.

## Suggested Test File Grouping
- `test_config.py`: settings precedence P0 cases.
- `test_cluster.py`: XYZ IO, fragmentation, sampling.
- `test_mbe.py`: subset generation, job_id, ghosts.
- `test_render.py`: input rendering (Q-Chem/ORCA).
- `test_run_control.py`: control parsing, logging/rename, confirm/retry/delete/state.
- `test_hpc_templates.py`: PBS/Slurm chunking, command wiring.
- `test_parsers.py`: auto-detect, metadata inference, field parsing.
- `test_analysis.py`: read_jsonl, group/aggregate, delta energy, export/plot.
- `test_cli.py`: fragment/gen/parse/analyze end-to-end smoke (can shell-invoke or simulate main).
