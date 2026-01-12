from __future__ import annotations

import os
try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10 fallback
    import tomli as tomllib  # type: ignore[import-not-found]
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


_ENV_MAP = {
    "qchem_command": "MBE_QCHEM_CMD",
    "orca_command": "MBE_ORCA_CMD",
    "qchem_module": "MBE_QCHEM_MODULE",
    "orca_module": "MBE_ORCA_MODULE",
    "scratch_dir": "MBE_SCRATCH",
    "scheduler_queue": "MBE_SCHED_QUEUE",
    "scheduler_partition": "MBE_SCHED_PARTITION",
    "scheduler_account": "MBE_SCHED_ACCOUNT",
}


@dataclass
class Settings:
    """Global defaults for mbe-tools CLI and APIs."""

    qchem_command: Optional[str] = None
    orca_command: Optional[str] = None
    qchem_module: Optional[str] = None
    orca_module: Optional[str] = None
    scratch_dir: Optional[str] = None

    scheduler_queue: Optional[str] = None
    scheduler_partition: Optional[str] = None
    scheduler_account: Optional[str] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Settings":
        known = {k: data.get(k) for k in _ENV_MAP.keys()}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(**known, extra=extra)

    def merged(self, other: Mapping[str, Any]) -> "Settings":
        data = self.to_dict()
        data.update(other)
        return Settings.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qchem_command": self.qchem_command,
            "orca_command": self.orca_command,
            "qchem_module": self.qchem_module,
            "orca_module": self.orca_module,
            "scratch_dir": self.scratch_dir,
            "scheduler_queue": self.scheduler_queue,
            "scheduler_partition": self.scheduler_partition,
            "scheduler_account": self.scheduler_account,
            **self.extra,
        }


_cached: Optional[Settings] = None
_stack: list[Settings] = []


def _read_toml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}


def _env_settings(env: Mapping[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, env_name in _ENV_MAP.items():
        if env_name in env:
            out[key] = env[env_name]
    return out


def load_settings(
    path: Optional[str] = None,
    *,
    overrides: Optional[Mapping[str, Any]] = None,
    env: Mapping[str, str] = os.environ,
    cwd: Optional[str] = None,
) -> Settings:
    """Load settings with precedence: explicit path > ./mbe.toml > ~/.config/mbe-tools/config.toml > env."""

    base = Settings()
    merged: Dict[str, Any] = {}

    # Lowest priority: environment variables
    merged.update(_env_settings(env))

    # User-global config (~/.config/mbe-tools/config.toml)
    home = os.path.expanduser("~")
    global_path = os.path.join(home, ".config", "mbe-tools", "config.toml")
    merged.update(_read_toml(global_path))

    # Project-local config (./mbe.toml)
    working_dir = cwd or os.getcwd()
    local_path = os.path.join(working_dir, "mbe.toml")
    merged.update(_read_toml(local_path))

    # Explicit config path
    if path:
        merged.update(_read_toml(path))

    if overrides:
        merged.update(dict(overrides))

    return base.merged(merged)


def get_settings() -> Settings:
    global _cached
    if _stack:
        return _stack[-1]
    if _cached is None:
        _cached = load_settings()
    return _cached


class use_settings:
    """Context manager to temporarily override global settings."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def __enter__(self) -> Settings:
        _stack.append(self.settings)
        return self.settings

    def __exit__(self, exc_type, exc, tb) -> None:
        if _stack:
            _stack.pop()
