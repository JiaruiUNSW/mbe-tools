"""mbe-tools: Many-Body Expansion workflow toolkit."""

DEFAULT_VERSION = "0.2.1"

try:  # pragma: no cover - defensive import
	from importlib.metadata import version, PackageNotFoundError  # type: ignore

	try:
		__version__ = version("mbe-tools")
	except PackageNotFoundError:
		__version__ = DEFAULT_VERSION
except Exception:  # pragma: no cover - keep import cheap
	__version__ = DEFAULT_VERSION

from .cluster import (
	read_xyz,
	write_xyz,
	fragment_by_water_heuristic,
	sample_fragments,
	spatial_sample_fragments,
	FragmentRecord,
)
from .mbe import MBEParams, generate_subsets_xyz
from .parsers.base import ParsedRecord
from .config import Settings, load_settings, get_settings, use_settings
from .mbe_math import assemble_mbe_energy
