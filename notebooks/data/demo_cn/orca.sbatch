#!/bin/bash
#SBATCH --job-name=mbe-orca
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --output=mbe-orca.%j.out
#SBATCH --error=mbe-orca.%j.err

set -euo pipefail
shopt -s nullglob

module load orca/5.0.3
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

FILES_PER_JOB=5
files=( *.inp )
if ((${#files[@]} == 0)); then echo '[ERR] no inputs'; exit 1; fi
job_index=0
start=0
while (( start < ${#files[@]} )); do
  job_index=$(( job_index + 1 ))
  chunk=()
  end=$(( start + FILES_PER_JOB ))
  if (( end > ${#files[@]} )); then end=${#files[@]}; fi
  for (( i=start; i<end; i++ )); do chunk+=( "${files[i]}" ); done
  jobname_chunk="${job_name}_${job_index}"
  sbatchfile="._${jobname_chunk}.sbatch"
  cat > "${sbatchfile}" <<EOF
#!/bin/bash
#SBATCH --job-name=${jobname_chunk}
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --output=${jobname_chunk}.%j.out
#SBATCH --error=${jobname_chunk}.%j.err

set -euo pipefail
shopt -s nullglob
module load orca/5.0.3
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
files_to_run=( ${chunk[@]} )
CPUS_PER_TASK=16
ORCA_CMD=${ORCA_CMD:-orca}
WRAPPER_LOG=${WRAPPER_LOG:-.mbe_wrapper.log}

run_with_control() {
  local input="$1"
  local ctrl=""
  if [ -f "${input}.mbe.control.toml" ]; then
    ctrl="${input}.mbe.control.toml"
  elif [ -f "mbe.control.toml" ]; then
    ctrl="mbe.control.toml"
  fi
  python - "${input}" "${ctrl}" orca "${ORCA_CMD}" "${CPUS_PER_TASK}" "${WRAPPER_LOG}" <<'PY'

import json
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
            w.write(msg + "
")
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
    print(f"[ERR] tomllib unavailable: {exc}", file=sys.stderr)
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

    if backend == "qchem":
        cmd = [cmd_bin, "-np", str(ncpus), input_path, str(log_tmp)]
        run_kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        log_handle = None
    else:
        log_handle = open(log_tmp, "w", encoding="utf-8")
        cmd = [cmd_bin, input_path]
        run_kwargs = {"stdout": log_handle, "stderr": subprocess.STDOUT}

    log_wrapper(f"[INFO] attempt {attempt}/{total_attempts} {input_path} -> {log_tmp}")
    result = subprocess.run(cmd, **run_kwargs)
    if log_handle is not None:
        log_handle.close()
    last_exit = result.returncode

    if not log_tmp.exists():
        log_wrapper(f"[WARN] missing log {log_tmp}")
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

PY
}

for f in "${files_to_run[@]}"; do
  [ -f "$f" ] || continue
  run_with_control "$f"
done
EOF
  sbatch "${sbatchfile}"
  start=$end
done

echo "[OK] submitted $job_index Slurm jobs"
