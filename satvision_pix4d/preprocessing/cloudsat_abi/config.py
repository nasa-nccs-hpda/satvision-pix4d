"""Configuration and satellite definitions for CloudSat-ABI collocation."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass


DEFAULT_GEOMETRY_DIR = Path(
    "/explore/nobackup/projects/pix4dcloud/jgong"
)


@dataclass(frozen=True)
class SatelliteSpec:
    """A GOES spacecraft and its operational viewing region."""

    token: str
    name: str
    region: str
    number: int

    @property
    def platform_code(self) -> str:
        return f"G{self.number}"

    @property
    def filename_token(self) -> str:
        return self.name.replace("-", "")

    @property
    def geometry_filename(self) -> str:
        return f"ABI_{self.region.upper()}_GEO_TOPO_LOMSK.nc"

    def geometry_path(self, directory: Path = DEFAULT_GEOMETRY_DIR) -> Path:
        return Path(directory) / self.geometry_filename


SATELLITES = {
    "goes16": SatelliteSpec("goes16", "GOES-16", "east", 16),
    "goes17": SatelliteSpec("goes17", "GOES-17", "west", 17),
    "goes18": SatelliteSpec("goes18", "GOES-18", "west", 18),
    "goes19": SatelliteSpec("goes19", "GOES-19", "east", 19),
}


def get_satellite(token: str) -> SatelliteSpec:
    key = token.lower().replace("-", "")
    try:
        return SATELLITES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported satellite {token!r}") from exc


@dataclass(frozen=True)
class CropConfig:
    """Validated settings for one collocation run."""

    abi_root: Path
    cloudsat_root: Path
    cloudsat_index_root: Path
    latlon_path: Path
    output_dir: Path
    year: int
    satellite: SatelliteSpec
    day_start: int = 1
    day_end: int | None = None
    orbit: str | None = None
    transect: tuple[float, float] = (-90.0, 90.0)
    offsets: tuple[int, ...] = (-40, -20, 0, 20, 40)
    chip_size: int = 128
    profile_stride: int = 45
    profiles_per_chip: int = 91
    metadata: frozenset[str] = frozenset({"cloudsat"})
    merra2_root: Path | None = None
    merra2_variables: tuple[str, ...] = ()
    max_scan_delta_minutes: float = 8.0
    allow_missing_timesteps: bool = False
    overwrite: bool = False
    max_chips: int | None = None

    def __post_init__(self):
        if self.chip_size <= 0 or self.chip_size % 2:
            raise ValueError("chip_size must be a positive even integer")
        if self.profile_stride <= 0:
            raise ValueError("profile_stride must be positive")
        if self.profiles_per_chip <= 0 or self.profiles_per_chip % 2 == 0:
            raise ValueError("profiles_per_chip must be a positive odd integer")
        if self.day_end is not None and self.day_end < self.day_start:
            raise ValueError("day_end must be greater than or equal to day_start")
        if not self.offsets:
            raise ValueError("at least one ABI offset is required")
        if "merra2" in self.metadata and self.merra2_root is None:
            raise ValueError("merra2_root is required when MERRA-2 metadata is enabled")
        unknown = self.metadata - {"cloudsat", "merra2"}
        if unknown:
            raise ValueError(f"Unsupported metadata groups: {sorted(unknown)}")

        low, high = sorted(self.transect)
        if low < -90 or high > 90:
            raise ValueError("transect latitude bounds must be within [-90, 90]")
        object.__setattr__(self, "transect", (float(low), float(high)))
