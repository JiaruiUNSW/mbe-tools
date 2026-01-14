from __future__ import annotations
import textwrap
from typing import Optional


_RUN_CONTROL_PY = textwrap.dedent(
    """
import json
import shlex
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

input_path, control_path, backend, cmd_bin, ncpus, wrapper_log = sys.argv[1:7]
base = Path(input_path)


def _deep_merge(default: dict, raw: dict) -> dict:
    merged = {k: (v.copy() if isinstance(v, dict) else v) for k, v in default.items()}
    for section in ("confirm", "retry", "delete", "state", "template"):
        if section in raw and isinstance(raw[section], dict):
            merged_section = merged.get(section, {}).copy()
            merged_section.update(raw[section])
            merged[section] = merged_section
    merged["version"] = raw.get("version", merged.get("version", 1))
    return merged


def load_control(path: str) -> tuple[dict, bool]:
    default = {
        "confirm": {"log_path": str(base.with_suffix("._try.out")), "regex_any": [], "regex_none": []},
        "retry": {
            "enabled": False,
            "max_attempts": 0,
            "sleep_seconds": 0,
            "cleanup_globs": [],
            "write_failed_last": False,
            "failed_last_path": str(base.with_suffix(".failed.out")),
        },
        "delete": {
            "enabled": False,
            "on_success_only": True,
            "delete_inputs_globs": [],
            "delete_outputs_globs": [],
            "allow_delete_outputs": False,
        },
        "state": {"state_file": ".mbe_state.json", "skip_if_done": True},
        "template": {"strict": False},
        "version": 1,
    }
    if not path:
        return default, False
    p = Path(path)
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        return {"_parse_error": str(exc), **default}, True
    if raw.get("version") != 1:
        return {"_parse_error": "missing_or_bad_version", **default}, True
    return _deep_merge(default, raw), True


def log_wrapper(msg: str) -> None:
    if not wrapper_log:
        return
    try:
        with open(wrapper_log, "a", encoding="utf-8") as w:
            w.write(msg + "\\n")
    except Exception:  # pragma: no cover - best effort
        pass


def read_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception:  # pragma: no cover - lenient
        return {}


def write_state(path: Path, key: str, entry: dict) -> None:
    try:
        current = read_state(path)
        current[key] = entry
        path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - best effort
        log_wrapper(f"[WARN] failed to write state: {exc}")


try:
    import tomllib  # type: ignore
except Exception as exc:  # pragma: no cover - tomllib should exist
    try:
        import tomli as tomllib  # type: ignore
    except Exception as exc2:  # pragma: no cover - fallback
        print(f"[ERR] tomllib unavailable: {exc}; tomli fallback failed: {exc2}", file=sys.stderr)
        sys.exit(2)

control, had_control = load_control(control_path)
if control.get("_parse_error"):
    msg = f"[WARN] control parse failed ({control['_parse_error']}); run-control disabled"
    if control.get("template", {}).get("strict"):
        print(msg, file=sys.stderr)
        sys.exit(3)
    log_wrapper(msg)
    control = load_control("")[0]
    had_control = False

state_enabled = had_control
state_file = Path(control["state"]["state_file"]) if state_enabled else None
state_key = base.name
if state_enabled and control["state"].get("skip_if_done"):
    st = read_state(state_file)
    if st.get(state_key, {}).get("status") == "done":
        log_wrapper(f"[INFO] skip {input_path}: already done")
        sys.exit(0)

retry_enabled = control["retry"]["enabled"]
max_attempts = control["retry"]["max_attempts"] if retry_enabled else 0
total_attempts = 1 + max_attempts
confirm_any = control["confirm"]["regex_any"]
confirm_none = control["confirm"]["regex_none"]
confirm_enabled = bool(confirm_any) and had_control
log_tmp_default = Path(control["confirm"].get("log_path") or base.with_suffix("._try.out"))
final_log = base.with_suffix(".out")
attempt_logs: list[str] = []
last_exit = 0
last_matches: list[str] = []
last_log_path: Path | None = None
success = False

for attempt in range(1, total_attempts + 1):
    log_tmp = log_tmp_default
    if not log_tmp.is_absolute():
        log_tmp = Path.cwd() / log_tmp
    if log_tmp.exists():
        try:
            log_tmp.unlink()
        except Exception:
            pass

    cmd_head = shlex.split(cmd_bin)
    if backend == "qchem":
        cmd = cmd_head + ["-np", str(ncpus), input_path, str(log_tmp)]
        run_kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        log_handle = None
    else:
        log_handle = open(log_tmp, "w", encoding="utf-8")
        cmd = cmd_head + [input_path]
        run_kwargs = {"stdout": log_handle, "stderr": subprocess.STDOUT}

    log_wrapper(f"[INFO] attempt {attempt}/{total_attempts} {input_path} -> {log_tmp}")
    result = subprocess.run(cmd, **run_kwargs)
    if log_handle is not None:
        log_handle.close()
    last_exit = result.returncode

    if not log_tmp.exists():
        log_wrapper(f"[WARN] missing log {log_tmp}")
        try:
            log_tmp.parent.mkdir(parents=True, exist_ok=True)
            log_tmp.write_text(
                f"[WRAPPER] Q-Chem produced no log file.\\ncmd={cmd}\\nreturncode={last_exit}\\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        matches_any: list[str] = []
        has_block = False
    else:
        text = log_tmp.read_text(encoding="utf-8", errors="ignore")
        matches_any = [r for r in confirm_any if re.search(r, text)] if confirm_any else []
        has_block = any(re.search(r, text) for r in confirm_none) if confirm_none else False

    if confirm_enabled:
        success = bool(matches_any) and not has_block and last_exit == 0
    else:
        success = last_exit == 0

    last_matches = matches_any
    last_log_path = log_tmp
    target = final_log if success else base.with_suffix(f".attempt{attempt}.out")
    if not log_tmp.exists():
        try:
            log_tmp.write_text("", encoding="utf-8")
        except Exception:
            pass
    if target.exists():
        try:
            target.unlink()
        except Exception:
            pass
    try:
        log_tmp.rename(target)
    except Exception:
        shutil.copy2(log_tmp, target)
        try:
            log_tmp.unlink()
        except Exception:
            pass
    attempt_logs.append(str(target))

    if success:
        break

    if attempt < total_attempts:
        for pat in control["retry"]["cleanup_globs"]:
            for candidate in Path(".").glob(pat):
                try:
                    if candidate.is_file():
                        candidate.unlink()
                    elif candidate.is_dir():
                        shutil.rmtree(candidate)
                except Exception:
                    pass
        sleep_s = control["retry"].get("sleep_seconds") or 0
        if sleep_s:
            time.sleep(float(sleep_s))

if not success and control["retry"].get("write_failed_last"):
    failed_alias = Path(control["retry"].get("failed_last_path") or base.with_suffix(".failed.out"))
    src = Path(attempt_logs[-1]) if attempt_logs else last_log_path
    if src and src.exists():
        try:
            shutil.copy2(src, failed_alias)
        except Exception:
            pass

if success and control["delete"]["enabled"]:
    if (not control["delete"].get("on_success_only", True)) or success:
        for pat in control["delete"].get("delete_inputs_globs", []):
            for candidate in Path(".").glob(pat):
                try:
                    if candidate.is_file():
                        candidate.unlink()
                except Exception:
                    pass
        if control["delete"].get("allow_delete_outputs") and control["delete"].get("delete_outputs_globs"):
            for pat in control["delete"]["delete_outputs_globs"]:
                for candidate in Path(".").glob(pat):
                    try:
                        if candidate.is_file():
                            candidate.unlink()
                    except Exception:
                        pass

if state_enabled:
    entry = {
        "version": 1,
        "status": "done" if success else "failed",
        "attempts": len(attempt_logs),
        "last_exit_code": last_exit,
        "confirmed": bool(last_matches) if confirm_enabled else None,
        "final_log": str(final_log if success else (Path(attempt_logs[-1]) if attempt_logs else last_log_path)),
        "attempt_logs": attempt_logs,
        "matched": last_matches,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }
    write_state(state_file, state_key, entry)

if not success:
    log_wrapper(f"[FAIL] {input_path} exit={last_exit} matches={last_matches}")
    sys.exit(last_exit or 1)

log_wrapper(f"[OK] {input_path} done")
sys.exit(0)
    """
)


def _run_with_control_block(backend: str, command_var: str, ncpus_var: str, default_command: str) -> list[str]:
    return [
        f"{command_var}=${{{command_var}:-{default_command}}}",
        "WRAPPER_LOG=${WRAPPER_LOG:-.mbe_wrapper.log}",
        "PY_BIN=${PY_BIN:-python3}",
        "",
        "run_with_control() {",
        "  local input=\"$1\"",
        "  local ctrl=\"\"",
        "  if [ -n \"${CTRL_FILE:-}\" ] && [ -f \"${CTRL_FILE}\" ]; then",
        "    ctrl=\"${CTRL_FILE}\"",
        "  elif [ -f \"${input}.mbe.control.toml\" ]; then",
        "    ctrl=\"${input}.mbe.control.toml\"",
        "  elif [ -f \"mbe.control.toml\" ]; then",
        "    ctrl=\"mbe.control.toml\"",
        "  fi",
        f"  ${{PY_BIN}} - \"${{input}}\" \"${{ctrl}}\" {backend} \"${{{command_var}}}\" \"${{{ncpus_var}}}\" \"${{WRAPPER_LOG}}\" <<'PY'",
        _RUN_CONTROL_PY,
        "PY",
        "}",
        "",
    ]


def _default_control_file_lines(path: str) -> list[str]:
    return [
        f"cat > \"{path}\" <<'EOF'",
        "version = 1",
        "",
        "[confirm]",
        "regex_any = [\"Thank you very much\", \"TOTAL ENERGY =\"]",
        "regex_none = [\"SCF failed\", \"UNCONVERGED\", \"Q-Chem fatal error\"]",
        "",
        "[retry]",
        "enabled = true",
        "max_attempts = 2",
        "sleep_seconds = 5",
        "cleanup_globs = [\"*.rwf\", \"qchem_scratch/*\"]",
        "write_failed_last = true",
        "failed_last_path = \"last_failed.out\"",
        "",
        "[delete]",
        "enabled = false",
        "on_success_only = true",
        "delete_inputs_globs = []",
        "delete_outputs_globs = []",
        "allow_delete_outputs = false",
        "",
        "[state]",
        "state_file = \".mbe_state.json\"",
        "skip_if_done = true",
        "",
        "[template]",
        "strict = false",
        "EOF",
        "",
    ]


def render_pbs_qchem(
    *,
    job_name: str,
    walltime: str = "24:00:00",
    ncpus: int = 16,
    mem_gb: float = 32.0,
    queue: Optional[str] = None,
    project: Optional[str] = None,
    module: str = "qchem/5.2.2",
    input_glob: str = "*.inp",
    chunk_size: Optional[int] = None,
    wrapper: bool = False,
    local_run: bool = False,
    control_file: Optional[str] = None,
    builtin_control: bool = False,
) -> str:
    if control_file:
        ctrl_path = control_file
        control_lines: list[str] = []
    elif builtin_control:
        ctrl_path = ".mbe_default.control.toml"
        control_lines = _default_control_file_lines(ctrl_path)
    else:
        ctrl_path = None
        control_lines = []
    ctrl_env = [f"CTRL_FILE={ctrl_path}"] if ctrl_path else []
    ctrl_env_heredoc = [f"CTRL_FILE=${{CTRL_FILE:-{ctrl_path}}}"] if ctrl_path else []
    mem_mb = int(mem_gb * 1000)
    if wrapper and chunk_size and chunk_size > 0:
        # Special-case: per-input submitter (one PBS per .inp)
        if chunk_size == 1:
            lines = [
                "#!/bin/bash",
                "set -euo pipefail",
                "shopt -s nullglob",
                "",
                *ctrl_env,
                "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"",
                "INPUT_DIR=${INPUT_DIR:-\"${SCRIPT_DIR}/inputs_qchem\"}",
                "ABS_INPUT_DIR=\"$(cd \"${INPUT_DIR}\" && pwd)\"",
                "cd \"${ABS_INPUT_DIR}\"",
                *control_lines,
                "",
                f"WALLTIME=${{WALLTIME:-{walltime}}}",
                f"MEM_MB=${{MEM_MB:-{mem_mb}}}",
                f"NCPUS=${{NCPUS:-{ncpus}}}",
                f"QC_MOD=${{QC_MOD:-{module}}}",
                "TMPDIR=${TMPDIR:-/tmp}",
                "FILES=( *.inp )",
                "if ((${#FILES[@]} == 0)); then echo '[ERR] no inputs'; exit 1; fi",
                "count=0",
                "for f in \"${FILES[@]}\"; do",
                "  [ -f \"$f\" ] || continue",
                "  base=\"${f%.inp}\"",
                "  jobname_raw=\"${base}\"",
                "  jobname_safe=${jobname_raw//[^A-Za-z0-9_]/_}",
                "  jobname_safe=${jobname_safe:0:15}",
                "  if [[ -z $jobname_safe ]]; then jobname_safe=m_job; fi",
                "  [[ $jobname_safe =~ ^[A-Za-z] ]] || jobname_safe=m_${jobname_safe}",
                "  pbsfile=\".pbs_${base}.pbs\"",
                "  logfile=\"${ABS_INPUT_DIR}/${base}.log\"",
                "  outfile=\"${ABS_INPUT_DIR}/${base}.out\"",
                "  input_path=\"${ABS_INPUT_DIR}/${f}\"",
                "  set +u",
                "  cat > \"${pbsfile}\" <<EOF",
                "#!/bin/bash",
                "#PBS -N ${jobname_safe}",
                "#PBS -l walltime=${WALLTIME},mem=${MEM_MB}Mb,ncpus=${NCPUS}",
                "#PBS -j oe",
                "#PBS -o ${logfile}",
                "",
                "cd \"${PBS_O_WORKDIR:-${ABS_INPUT_DIR}}\"",
                "TMPDIR=${TMPDIR:-/tmp/qchem_${USER:-user}}",
                "mkdir -p \"${TMPDIR}\" || true",
                "export TMPDIR",
                "QCSCRATCH=${QCSCRATCH:-${TMPDIR}/qcscratch_${PBS_JOBID:-$$}}",
                "mkdir -p \"${QCSCRATCH}\" || true",
                "export QCSCRATCH",
                "module load ${QC_MOD}",
                "",
                "echo \"[INFO] $(date) Start Q-Chem: ${input_path} -> ${outfile} (np=${NCPUS})\"",
                "qchem -np ${NCPUS} \"${input_path}\" \"${outfile}\" \"${QCSCRATCH}\"",
                "echo \"[INFO] $(date) Finished: ${input_path}\"",
                "EOF",
                "  set -u",
                "  qsub \"${pbsfile}\"",
                "  count=$((count + 1))",
                "  sleep 0.2",
                "done",
                "echo \"[OK] submitted ${count} PBS jobs (one per .inp)\"",
            ]
            return "\n".join(lines) + "\n"

        lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            "shopt -s nullglob",
            "",
            "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"",
            "INPUT_DIR=${INPUT_DIR:-\"${SCRIPT_DIR}/inputs_qchem\"}",
            "cd \"${INPUT_DIR}\"",
            "",
            *ctrl_env,
            *control_lines,
            f"JOB_NAME=${{JOB_NAME:-{job_name}}}",
            "SANITIZED_JOB_NAME=${JOB_NAME//[^A-Za-z0-9_]/_}",
            "if [[ ! ${SANITIZED_JOB_NAME} =~ ^[A-Za-z] ]]; then SANITIZED_JOB_NAME=m_${SANITIZED_JOB_NAME}; fi",
            f"WALLTIME=${{WALLTIME:-{walltime}}}",
            f"MEM_MB=${{MEM_MB:-{mem_mb}}}",
            f"NCPUS=${{NCPUS:-{ncpus}}}",
            f"QUEUE=${{QUEUE:-{queue or ''}}}",
            f"PROJECT=${{PROJECT:-{project or ''}}}",
            f"QC_MOD=${{QC_MOD:-{module}}}",
            f"FILES_PER_JOB=${{FILES_PER_JOB:-{chunk_size}}}",
            f"FILES=( {input_glob} )",
            "if ((${#FILES[@]} == 0)); then echo '[ERR] no inputs'; exit 1; fi",
            "job_index=0",
            "start=0",
            "while (( start < ${#FILES[@]} )); do",
            "  job_index=$(( job_index + 1 ))",
            "  chunk=()",
            "  end=$(( start + FILES_PER_JOB ))",
            "  if (( end > ${#FILES[@]} )); then end=${#FILES[@]}; fi",
            "  for (( i=start; i<end; i++ )); do chunk+=( \"${FILES[i]}\" ); done",
            "  jobname=\"${SANITIZED_JOB_NAME}_${job_index}\"",
            "  pbsfile=\"._${jobname}.pbs\"",
            "  # Build escaped chunk for safe heredoc serialization",
            "  chunk_escaped=\"\"",
            "  for f in \"${chunk[@]}\"; do",
            "    chunk_escaped+=\" $(printf '%q' \"$f\")\"",
            "  done",
            "  set +u",
            "  cat > \"${pbsfile}\" <<EOF",
            "#!/bin/bash",
            "#PBS -j oe",
            "#PBS -N ${jobname}",
            "#PBS -l walltime=${WALLTIME},mem=${MEM_MB}Mb,ncpus=${NCPUS}",
            "#PBS -o ${jobname}.log",
        ]
        if queue:
            lines.append("#PBS -q ${QUEUE}")
        if project:
            lines.append("#PBS -P ${PROJECT}")
        lines += [
            "",
            "cd \"${PBS_O_WORKDIR:-.}\"",
            *ctrl_env_heredoc,
            "export TMPDIR=${TMPDIR:-/tmp}",
            "module load ${QC_MOD}",
            "eval \"files_to_run=(${chunk_escaped})\"",
            "NCPUS=${NCPUS}",
            "# Default command: qchem -np ${NCPUS}",
            *_run_with_control_block("qchem", "QC_CMD", "NCPUS", "qchem"),
            "for f in \"${files_to_run[@]}\"; do",
            "  [ -f \"$f\" ] || continue",
            "  run_with_control \"$f\"",
            "done",
            "EOF",
            "  set -u",
            "  qsub \"${pbsfile}\"",
            "  start=$end",
            "done",
            "",
            "echo \"[OK] submitted $job_index PBS jobs\"",
        ]
        return "\n".join(lines) + "\n"

    if wrapper and not chunk_size:
        lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            "shopt -s nullglob",
            "",
            *ctrl_env,
            *control_lines,
            f"JOB_NAME=${{JOB_NAME:-{job_name}}}",
            f"WALLTIME=${{WALLTIME:-{walltime}}}",
            f"MEM_MB=${{MEM_MB:-{mem_mb}}}",
            f"NCPUS=${{NCPUS:-{ncpus}}}",
            f"QUEUE=${{QUEUE:-{queue or ''}}}",
            f"PROJECT=${{PROJECT:-{project or ''}}}",
            f"QC_MOD=${{QC_MOD:-{module}}}",
            f"PBSFILE=${{PBSFILE:-._{job_name}.pbs}}",
            "",
            "cat > \"${PBSFILE}\" <<EOF",
            "#!/bin/bash",
            "#PBS -N ${JOB_NAME}",
            "#PBS -l walltime=${WALLTIME},mem=${MEM_MB}Mb,ncpus=${NCPUS}",
            "#PBS -j oe",
            "#PBS -o ${JOB_NAME}.log",
        ]
        if queue:
            lines.append("#PBS -q ${QUEUE}")
        if project:
            lines.append("#PBS -P ${PROJECT}")
        lines += [
            "",
            "set -euo pipefail",
            "shopt -s nullglob",
            "cd \"${PBS_O_WORKDIR:-.}\"",
            "export TMPDIR=${TMPDIR:-/tmp}",
            "module load ${QC_MOD}",
            "",
            "NCPUS=${NCPUS}",
            "# Default command: qchem -np ${NCPUS}",
        ]
        lines += _run_with_control_block("qchem", "QC_CMD", "NCPUS", "qchem")
        lines += [
            f"files=( {input_glob} )",
            "if ((${#files[@]} == 0)); then echo '[ERR] no inputs'; exit 1; fi",
            "",
            "for f in \"${files[@]}\"; do",
            "  [ -f \"$f\" ] || continue",
            "  run_with_control \"$f\"",
            "done",
            "",
            "echo \"[OK] all done\"",
            "EOF",
            "",
            "qsub \"${PBSFILE}\"",
            "echo \"[OK] submitted ${PBSFILE}\"",
        ]
        return "\n".join(lines) + "\n"

    if chunk_size and chunk_size > 0:
        if local_run:
            lines = [
                "#!/bin/bash",
                "set -euo pipefail",
                "shopt -s nullglob",
                "",
                *ctrl_env,
                *control_lines,
                f"FILES_PER_JOB={chunk_size}",
                f"MEM={mem_mb}Mb",
                f"NCPUS={ncpus}",
                f"WALLTIME={walltime}",
                f"QC_MOD={module}",
                f"BASE_JOBNAME={job_name}",
                "",
                "if command -v module >/dev/null 2>&1; then",
                "  module load python3/3.11.7 || true",
                "  module load ${QC_MOD} || true",
                "fi",
                "",
                *_run_with_control_block("qchem", "QC_CMD", "NCPUS", "qchem"),
                "",
                f"files=( {input_glob} )",
                "if ((${#files[@]} == 0)); then echo '[ERR] no inputs'; exit 1; fi",
                "job_index=0",
                "start=0",
                "while (( start < ${#files[@]} )); do",
                "  job_index=$(( job_index + 1 ))",
                "  chunk=()",
                "  end=$(( start + FILES_PER_JOB ))",
                "  if (( end > ${#files[@]} )); then end=${#files[@]}; fi",
                "  for (( i=start; i<end; i++ )); do chunk+=( \"${files[i]}\" ); done",
                f"  NCPUS={ncpus}",
                "  for f in \"${chunk[@]}\"; do",
                "    [ -f \"$f\" ] || continue",
                "    run_with_control \"$f\" || echo \"[WARN] failed: $f\"",
                "  done",
                "  start=$end",
                "done",
                "",
                "echo \"[OK] processed $job_index chunks locally\"",
            ]
            return "\n".join(lines) + "\n"

        lines = [
            "#!/bin/bash",
            f"#PBS -N {job_name}",
            "#PBS -j oe",
            f"#PBS -l walltime={walltime},mem={mem_mb}Mb,ncpus={ncpus}",
            f"#PBS -o {job_name}.log",
        ]
        if queue:
            lines.append(f"#PBS -q {queue}")
        if project:
            lines.append(f"#PBS -P {project}")
        lines += [
            "",
            "set -euo pipefail",
            "shopt -s nullglob",
            "",
            *ctrl_env,
            *control_lines,
            f"FILES_PER_JOB={chunk_size}",
            f"MEM={mem_mb}Mb",
            f"NCPUS={ncpus}",
            f"WALLTIME={walltime}",
            f"QC_MOD={module}",
            f"BASE_JOBNAME={job_name}",
            "",
            f"files=( {input_glob} )",
            "if ((${#files[@]} == 0)); then echo '[ERR] no inputs'; exit 1; fi",
            "job_index=0",
            "start=0",
            "while (( start < ${#files[@]} )); do",
            "  job_index=$(( job_index + 1 ))",
            "  chunk=()",
            "  end=$(( start + FILES_PER_JOB ))",
            "  if (( end > ${#files[@]} )); then end=${#files[@]}; fi",
            "  for (( i=start; i<end; i++ )); do chunk+=( \"${files[i]}\" ); done",
            "  jobname=\"${BASE_JOBNAME}_${job_index}\"",
            "  pbsfile=\"._${jobname}.pbs\"",
            "  # Build escaped chunk for safe heredoc serialization",
            "  chunk_escaped=\"\"",
            "  for f in \"${chunk[@]}\"; do",
            "    chunk_escaped+=\" $(printf '%q' \"$f\")\"",
            "  done",
            "  cat > \"${pbsfile}\" <<EOF",
            "#!/bin/bash",
            "#PBS -j oe",
            "#PBS -N ${jobname}",
            "#PBS -l walltime=${WALLTIME},mem=${MEM},ncpus=${NCPUS}",
            "#PBS -o ${jobname}.log",
        ]
        if queue:
            lines.append("#PBS -q ${QUEUE:-" + queue + "}")
        if project:
            lines.append("#PBS -P ${PROJECT:-" + project + "}")
        lines += [
            "",
            "cd \"${PBS_O_WORKDIR:-.}\"",
            *ctrl_env_heredoc,
            "export TMPDIR=${TMPDIR:-/tmp}",
            "module load ${QC_MOD}",
            "eval \"files_to_run=(${chunk_escaped})\"",
            f"NCPUS={ncpus}",
            "# Default command: qchem -np ${NCPUS}",
            *_run_with_control_block("qchem", "QC_CMD", "NCPUS", "qchem"),
            "for f in \"${files_to_run[@]}\"; do",
            "  [ -f \"$f\" ] || continue",
            "  run_with_control \"$f\"",
            "done",
            "EOF",
            "  qsub \"${pbsfile}\"",
            "  start=$end",
            "done",
            "",
            "echo \"[OK] submitted $job_index PBS jobs\"",
        ]
        return "\n".join(lines) + "\n"

    if local_run:
        lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            "shopt -s nullglob",
            "",
            *ctrl_env,
            *control_lines,
            f"QC_MOD={module}",
            "QC_CMD=qchem",
            f"NCPUS={ncpus}",
            f"MEM_MB={mem_mb}",
            "# Default command: qchem -np ${NCPUS}",
            *_run_with_control_block("qchem", "QC_CMD", "NCPUS", "qchem"),
            "",
            "if command -v module >/dev/null 2>&1; then",
            "  module load python3/3.11.7 || true",
            "  module load ${QC_MOD} || true",
            "fi",
            f"for f in {input_glob}; do",
            "  [ -f \"$f\" ] || continue",
            "  run_with_control \"$f\" || echo \"[WARN] failed: $f\"",
            "done",
            "",
            "echo \"[OK] all done locally\"",
        ]
        return "\n".join(lines) + "\n"

    lines = [
        "#!/bin/bash",
        f"#PBS -N {job_name}",
        f"#PBS -l walltime={walltime},mem={mem_mb}Mb,ncpus={ncpus}",
        "#PBS -j oe",
        f"#PBS -o {job_name}.log",
    ]
    if queue:
        lines.append(f"#PBS -q {queue}")
    if project:
        lines.append(f"#PBS -P {project}")

    lines += [
        "",
        "set -euo pipefail",
        "shopt -s nullglob",
        "",
        *ctrl_env,
        *control_lines,
        "cd \"${PBS_O_WORKDIR:-.}\"",
        "export TMPDIR=${TMPDIR:-/tmp}",
        f"module load {module}",
        "",
        f"NCPUS={ncpus}",
        f"# Default command: qchem -np {ncpus}",
    ]
    lines += _run_with_control_block("qchem", "QC_CMD", "NCPUS", "qchem")
    lines += [
        f"files=( {input_glob} )",
        "if ((${#files[@]} == 0)); then echo '[ERR] no inputs'; exit 1; fi",
        "",
        "for f in \"${files[@]}\"; do",
        "  [ -f \"$f\" ] || continue",
        "  run_with_control \"$f\"",
        "done",
        "",
        "echo \"[OK] all done\"",
    ]
    return "\n".join(lines) + "\n"


def render_slurm_orca(
    *,
    job_name: str,
    walltime: str = "24:00:00",
    ntasks: int = 1,
    cpus_per_task: int = 16,
    mem_gb: float = 32.0,
    partition: Optional[str] = None,
    account: Optional[str] = None,
    qos: Optional[str] = None,
    module: str = "orca/5.0.3",
    command: str = "orca",
    input_glob: str = "*.inp",
    chunk_size: Optional[int] = None,
    wrapper: bool = False,
) -> str:
    mem_spec = f"{mem_gb:.2f}GB" if mem_gb % 1 else f"{int(mem_gb)}GB"
    if wrapper and chunk_size and chunk_size > 0:
        lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            "shopt -s nullglob",
            "",
            f"JOB_NAME=${{JOB_NAME:-{job_name}}}",
            f"WALLTIME=${{WALLTIME:-{walltime}}}",
            f"NTASKS=${{NTASKS:-{ntasks}}}",
            f"CPUS_PER_TASK=${{CPUS_PER_TASK:-{cpus_per_task}}}",
            f"MEM_SPEC=${{MEM_SPEC:-{mem_spec}}}",
            f"PARTITION=${{PARTITION:-{partition or ''}}}",
            f"ACCOUNT=${{ACCOUNT:-{account or ''}}}",
            f"QOS=${{QOS:-{qos or ''}}}",
            f"ORCA_MOD=${{ORCA_MOD:-{module}}}",
            f"ORCA_CMD=${{ORCA_CMD:-{command}}}",
            f"FILES_PER_JOB=${{FILES_PER_JOB:-{chunk_size}}}",
            f"FILES=( {input_glob} )",
            "if ((${#FILES[@]} == 0)); then echo '[ERR] no inputs'; exit 1; fi",
            "job_index=0",
            "start=0",
            "while (( start < ${#FILES[@]} )); do",
            "  job_index=$(( job_index + 1 ))",
            "  chunk=()",
            "  end=$(( start + FILES_PER_JOB ))",
            "  if (( end > ${#FILES[@]} )); then end=${#FILES[@]}; fi",
            "  for (( i=start; i<end; i++ )); do chunk+=( \"${FILES[i]}\" ); done",
            "  jobname=\"${JOB_NAME}_${job_index}\"",
            "  sbfile=\"._${jobname}.sbatch\"",
            "  # Build escaped chunk for safe heredoc serialization",
            "  chunk_escaped=\"\"",
            "  for f in \"${chunk[@]}\"; do",
            "    chunk_escaped+=\" $(printf '%q' \"$f\")\"",
            "  done",
            "  cat > \"${sbfile}\" <<EOF",
            "#!/bin/bash",
            "#SBATCH --job-name=${jobname}",
            "#SBATCH --time=${WALLTIME}",
            "#SBATCH --ntasks=${NTASKS}",
            "#SBATCH --cpus-per-task=${CPUS_PER_TASK}",
            "#SBATCH --mem=${MEM_SPEC}",
        ]
        if partition:
            lines.append("#SBATCH --partition=${PARTITION}")
        if account:
            lines.append("#SBATCH --account=${ACCOUNT}")
        if qos:
            lines.append("#SBATCH --qos=${QOS}")
        lines += [
            "#SBATCH --output=${jobname}.%j.out",
            "#SBATCH --error=${jobname}.%j.err",
            "",
            "set -euo pipefail",
            "shopt -s nullglob",
            "module load ${ORCA_MOD}",
            "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-${CPUS_PER_TASK}}",
            "eval \"files_to_run=(${chunk_escaped})\"",
            *_run_with_control_block("orca", "ORCA_CMD", "CPUS_PER_TASK", command),
            "for f in \"${files_to_run[@]}\"; do",
            "  [ -f \"$f\" ] || continue",
            "  run_with_control \"$f\"",
            "done",
            "EOF",
            "  sbatch \"${sbfile}\"",
            "  start=$end",
            "done",
            "",
            "echo \"[OK] submitted $job_index Slurm jobs\"",
        ]
        return "\n".join(lines) + "\n"

    if wrapper and not chunk_size:
        lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            "shopt -s nullglob",
            "",
            f"JOB_NAME=${{JOB_NAME:-{job_name}}}",
            f"WALLTIME=${{WALLTIME:-{walltime}}}",
            f"NTASKS=${{NTASKS:-{ntasks}}}",
            f"CPUS_PER_TASK=${{CPUS_PER_TASK:-{cpus_per_task}}}",
            f"MEM_SPEC=${{MEM_SPEC:-{mem_spec}}}",
            f"PARTITION=${{PARTITION:-{partition or ''}}}",
            f"ACCOUNT=${{ACCOUNT:-{account or ''}}}",
            f"QOS=${{QOS:-{qos or ''}}}",
            f"ORCA_MOD=${{ORCA_MOD:-{module}}}",
            f"ORCA_CMD=${{ORCA_CMD:-{command}}}",
            f"SBFILE=${{SBFILE:-._{job_name}.sbatch}}",
            "",
            "cat > \"${SBFILE}\" <<EOF",
            "#!/bin/bash",
            "#SBATCH --job-name=${JOB_NAME}",
            "#SBATCH --time=${WALLTIME}",
            "#SBATCH --ntasks=${NTASKS}",
            "#SBATCH --cpus-per-task=${CPUS_PER_TASK}",
            "#SBATCH --mem=${MEM_SPEC}",
        ]
        if partition:
            lines.append("#SBATCH --partition=${PARTITION}")
        if account:
            lines.append("#SBATCH --account=${ACCOUNT}")
        if qos:
            lines.append("#SBATCH --qos=${QOS}")
        lines += [
            "#SBATCH --output=${JOB_NAME}.%j.out",
            "#SBATCH --error=${JOB_NAME}.%j.err",
            "",
            "set -euo pipefail",
            "shopt -s nullglob",
            "module load ${ORCA_MOD}",
            "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-${CPUS_PER_TASK}}",
            "",
            "CPUS_PER_TASK=${CPUS_PER_TASK}",
        ]
        lines += _run_with_control_block("orca", "ORCA_CMD", "CPUS_PER_TASK", command)
        lines += [
            f"files=( {input_glob} )",
            "if ((${#files[@]} == 0)); then echo '[ERR] no inputs'; exit 1; fi",
            "",
            "for f in \"${files[@]}\"; do",
            "  [ -f \"$f\" ] || continue",
            "  run_with_control \"$f\"",
            "done",
            "",
            "echo \"[OK] all done\"",
            "EOF",
            "",
            "sbatch \"${SBFILE}\"",
            "echo \"[OK] submitted ${SBFILE}\"",
        ]
        return "\n".join(lines) + "\n"

    if chunk_size and chunk_size > 0:
        lines = [
            "#!/bin/bash",
            "#SBATCH --job-name=" + job_name,
            "#SBATCH --time=" + walltime,
            f"#SBATCH --ntasks={ntasks}",
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            f"#SBATCH --mem={mem_spec}",
        ]
        if partition:
            lines.append(f"#SBATCH --partition={partition}")
        if account:
            lines.append(f"#SBATCH --account={account}")
        if qos:
            lines.append(f"#SBATCH --qos={qos}")
        lines.append(f"#SBATCH --output={job_name}.%j.out")
        lines.append(f"#SBATCH --error={job_name}.%j.err")
        lines += [
            "",
            "set -euo pipefail",
            "shopt -s nullglob",
            "",
            f"module load {module}",
            "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-" + str(cpus_per_task) + "}",
            "",
            f"FILES_PER_JOB={chunk_size}",
            f"files=( {input_glob} )",
            "if ((${#files[@]} == 0)); then echo '[ERR] no inputs'; exit 1; fi",
            "job_index=0",
            "start=0",
            "while (( start < ${#files[@]} )); do",
            "  job_index=$(( job_index + 1 ))",
            "  chunk=()",
            "  end=$(( start + FILES_PER_JOB ))",
            "  if (( end > ${#files[@]} )); then end=${#files[@]}; fi",
            "  for (( i=start; i<end; i++ )); do chunk+=( \"${files[i]}\" ); done",
            "  jobname_chunk=\"${job_name}_${job_index}\"",
            "  sbatchfile=\"._${jobname_chunk}.sbatch\"",
            "  # Build escaped chunk for safe heredoc serialization",
            "  chunk_escaped=\"\"",
            "  for f in \"${chunk[@]}\"; do",
            "    chunk_escaped+=\" $(printf '%q' \"$f\")\"",
            "  done",
            "  cat > \"${sbatchfile}\" <<EOF",
            "#!/bin/bash",
            "#SBATCH --job-name=${jobname_chunk}",
            "#SBATCH --time=" + walltime,
            f"#SBATCH --ntasks={ntasks}",
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            f"#SBATCH --mem={mem_spec}",
        ]
        if partition:
            lines.append("#SBATCH --partition=" + partition)
        if account:
            lines.append("#SBATCH --account=" + account)
        if qos:
            lines.append("#SBATCH --qos=" + qos)
        lines += [
            "#SBATCH --output=${jobname_chunk}.%j.out",
            "#SBATCH --error=${jobname_chunk}.%j.err",
            "",
            "set -euo pipefail",
            "shopt -s nullglob",
            "module load " + module,
            "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-" + str(cpus_per_task) + "}",
            "eval \"files_to_run=(${chunk_escaped})\"",
            f"CPUS_PER_TASK={cpus_per_task}",
            *_run_with_control_block("orca", "ORCA_CMD", "CPUS_PER_TASK", command),
            "for f in \"${files_to_run[@]}\"; do",
            "  [ -f \"$f\" ] || continue",
            "  run_with_control \"$f\"",
            "done",
            "EOF",
            "  sbatch \"${sbatchfile}\"",
            "  start=$end",
            "done",
            "",
            "echo \"[OK] submitted $job_index Slurm jobs\"",
        ]
        return "\n".join(lines) + "\n"

    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=" + job_name,
        "#SBATCH --time=" + walltime,
        f"#SBATCH --ntasks={ntasks}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --mem={mem_spec}",
    ]
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    if account:
        lines.append(f"#SBATCH --account={account}")
    if qos:
        lines.append(f"#SBATCH --qos={qos}")
    lines.append(f"#SBATCH --output={job_name}.%j.out")
    lines.append(f"#SBATCH --error={job_name}.%j.err")

    lines += [
        "",
        "set -euo pipefail",
        "shopt -s nullglob",
        "",
        f"module load {module}",
        "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-" + str(cpus_per_task) + "}",
        "",
        f"CPUS_PER_TASK={cpus_per_task}",
    ]
    lines += _run_with_control_block("orca", "ORCA_CMD", "CPUS_PER_TASK", command)
    lines += [
        f"files=( {input_glob} )",
        "if ((${#files[@]} == 0)); then echo '[ERR] no inputs'; exit 1; fi",
        "",
        "for f in \"${files[@]}\"; do",
        "  [ -f \"$f\" ] || continue",
        "  run_with_control \"$f\"",
        "done",
        "",
        "echo \"[OK] all done\"",
    ]
    return "\n".join(lines) + "\n"
