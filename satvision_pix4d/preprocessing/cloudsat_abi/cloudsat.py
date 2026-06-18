"""CloudSat 2B-CLDCLASS-LIDAR transect access."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from satvision_pix4d.preprocessing.cloudsat_abi.utils import require_pyhdf


ORBIT_RE = re.compile(r"_(?P<orbit>\d{5})_")


@dataclass(frozen=True)
class CloudSatOrbitFile:
    day: int
    orbit: str
    path: Path


@dataclass
class CloudSatTransect:
    """CloudSat profiles restricted to a latitude transect."""

    source: Path
    latitude: np.ndarray
    longitude: np.ndarray
    utc_hour: np.ndarray
    quality: np.ndarray
    cloud_layer_base: np.ndarray
    cloud_layer_top: np.ndarray
    cloud_layer_type: np.ndarray
    _cached_cloud_mask: np.ndarray | None = field(
        default=None, init=False, repr=False
    )

    def __len__(self) -> int:
        return len(self.latitude)

    def profile_window(self, center: int, count: int) -> np.ndarray:
        start = center - count // 2
        stop = start + count
        if start < 0 or stop > len(self):
            raise IndexError("CloudSat profile window extends outside the transect")
        return np.arange(start, stop)

    def cloud_mask(self, levels: int = 40, resolution_km: float = 0.5) -> np.ndarray:
        if (
            levels == 40
            and resolution_km == 0.5
            and self._cached_cloud_mask is not None
        ):
            return self._cached_cloud_mask
        mask = np.zeros((len(self), levels), dtype=np.int8)
        for profile in range(len(self)):
            for layer in range(self.cloud_layer_base.shape[1]):
                base = self.cloud_layer_base[profile, layer]
                if base < 0:
                    continue
                top = self.cloud_layer_top[profile, layer]
                start = max(0, int(np.floor(base / resolution_km)))
                stop = min(levels, int(np.floor(top / resolution_km)) + 1)
                if stop > start:
                    mask[profile, start:stop] = self.cloud_layer_type[profile, layer]
        if levels == 40 and resolution_km == 0.5:
            self._cached_cloud_mask = mask
        return mask

    def metadata_arrays(self, center: int, count: int) -> dict[str, np.ndarray]:
        indices = self.profile_window(center, count)
        mask = self.cloud_mask()[indices]
        result = {
            "latitude": self.latitude[indices],
            "longitude": self.longitude[indices],
            "utc_hour": self.utc_hour[indices],
            "quality": self.quality[indices],
            "cloud_layer_base": self.cloud_layer_base[indices],
            "cloud_layer_top": self.cloud_layer_top[indices],
            "cloud_layer_type": self.cloud_layer_type[indices],
            "cloud_mask": mask,
            "cloud_mask_binary": (mask != 0).astype(np.int8),
        }
        return {f"cloudsat_{name}": value for name, value in result.items()}


class CloudSatReader:
    """Discover and read local CloudSat cloud-classification products."""

    def __init__(self, root: Path, index_root: Path | None = None):
        self.root = Path(root)
        self.index_root = Path(index_root or self.root / "2B-CLDCLASS-LIDAR")

    def discover_orbits(
        self,
        year: int,
        day_start: int = 1,
        day_end: int | None = None,
        orbit: str | None = None,
    ) -> Iterable[CloudSatOrbitFile]:
        year_root = self.index_root / str(year)
        if not year_root.is_dir():
            raise FileNotFoundError(
                f"CloudSat year directory does not exist: {year_root}"
            )
        final_day = day_end if day_end is not None else 366
        seen: set[tuple[int, str]] = set()
        for day_dir in sorted(year_root.iterdir()):
            if not day_dir.is_dir() or not day_dir.name.isdigit():
                continue
            day = int(day_dir.name)
            if not day_start <= day <= final_day:
                continue
            for path in sorted(day_dir.glob("*.hdf")):
                match = ORBIT_RE.search(path.name)
                if match is None:
                    continue
                orbit_id = match.group("orbit")
                key = (day, orbit_id)
                if key in seen or (orbit is not None and orbit_id != orbit):
                    continue
                seen.add(key)
                yield CloudSatOrbitFile(day, orbit_id, path)

    def read(
        self, path: Path, latitude_bounds: tuple[float, float]
    ) -> CloudSatTransect:
        HDF, SD, SDC = require_pyhdf()
        sd = SD(str(path), SDC.READ)
        try:
            arrays = {
                "cloud_layer_base": np.asarray(sd.select("CloudLayerBase")[:]),
                "cloud_layer_top": np.asarray(sd.select("CloudLayerTop")[:]),
                "cloud_layer_type": np.asarray(sd.select("CloudLayerType")[:]),
            }
        finally:
            sd.end()

        hdf = HDF(str(path), SDC.READ)
        vs = hdf.vstart()
        try:
            arrays.update(
                quality=self._read_vdata(vs, "Data_quality"),
                latitude=self._read_vdata(vs, "Latitude"),
                longitude=self._read_vdata(vs, "Longitude"),
                profile_time=self._read_vdata(vs, "Profile_time"),
                utc_start=self._read_vdata(vs, "UTC_start"),
            )
        finally:
            vs.end()
            hdf.close()

        low, high = latitude_bounds
        indices = np.flatnonzero(
            (arrays["latitude"] >= low) & (arrays["latitude"] <= high)
        )
        utc_hour = (
            arrays.pop("utc_start")[indices]
            + arrays.pop("profile_time")[indices]
        ) / 3600.0
        selected = {name: value[indices] for name, value in arrays.items()}
        return CloudSatTransect(source=Path(path), utc_hour=utc_hour, **selected)

    @staticmethod
    def _read_vdata(vs: Any, name: str) -> np.ndarray:
        handle = vs.attach(name)
        try:
            return np.squeeze(np.asarray(handle[:]))
        finally:
            handle.detach()
