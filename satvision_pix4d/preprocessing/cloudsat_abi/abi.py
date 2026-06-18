"""Local GOES ABI archive and geometry access."""

from __future__ import annotations

import re
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from satvision_pix4d.preprocessing.cloudsat_abi.config import SatelliteSpec
from satvision_pix4d.preprocessing.cloudsat_abi.utils import (
    datetime_from_year_doy,
    normalize_longitude,
    require_netcdf4,
)


LOG = logging.getLogger(__name__)

ABI_FILENAME_RE = re.compile(
    r"C(?P<channel>\d{2})_(?P<platform>G\d{2})_s"
    r"(?P<year>\d{4})(?P<doy>\d{3})(?P<hour>\d{2})"
    r"(?P<minute>\d{2})(?P<second>\d{2})"
)


@dataclass(frozen=True)
class ABIFileInfo:
    timestamp: datetime
    channel: int
    platform: str

    @classmethod
    def from_path(cls, path: str | Path) -> "ABIFileInfo | None":
        match = ABI_FILENAME_RE.search(Path(path).name)
        if match is None:
            return None
        timestamp = datetime_from_year_doy(
            match.group("year"), match.group("doy"), int(match.group("hour"))
        ) + timedelta(
            minutes=int(match.group("minute")),
            seconds=int(match.group("second")),
        )
        return cls(
            timestamp=timestamp,
            channel=int(match.group("channel")),
            platform=match.group("platform"),
        )


class ABIGeometry:
    """Find the nearest pixel in an East or West ABI geolocation grid."""

    def __init__(self, path: Path, coarse_target_size: int = 256):
        self.path = Path(path)
        self.coarse_target_size = coarse_target_size
        nc = require_netcdf4()
        with nc.Dataset(self.path) as dataset:
            self.latitude = np.asarray(
                dataset.variables["Latitude"][:], dtype=np.float32
            )
            self.longitude = np.asarray(
                dataset.variables["Longitude"][:], dtype=np.float32
            )

        invalid = (~np.isfinite(self.latitude)) | (~np.isfinite(self.longitude))
        invalid |= (np.abs(self.latitude) > 90) | (np.abs(self.longitude) > 360)
        self.valid = ~invalid
        if not np.any(self.valid):
            raise ValueError(f"No valid coordinates in {self.path}")
        self.use_360 = bool(np.nanmax(self.longitude[self.valid]) > 180)
        self.longitude = normalize_longitude(
            self.longitude, self.use_360
        ).astype(np.float32)
        self.lat_min = float(np.min(self.latitude[self.valid]))
        self.lat_max = float(np.max(self.latitude[self.valid]))

    def nearest(self, latitude: float, longitude: float) -> tuple[int, int]:
        longitude = float(normalize_longitude(longitude, self.use_360))
        if not self.lat_min <= latitude <= self.lat_max:
            raise ValueError(f"Latitude {latitude:.3f} is outside ABI coverage")

        stride = max(1, min(self.latitude.shape) // self.coarse_target_size)
        coarse_lat = self.latitude[::stride, ::stride]
        coarse_lon = self.longitude[::stride, ::stride]
        coarse_valid = self.valid[::stride, ::stride]
        distance = self._distance(coarse_lat, coarse_lon, latitude, longitude)
        distance[~coarse_valid] = np.inf
        coarse_row, coarse_column = np.unravel_index(
            int(np.argmin(distance)), distance.shape
        )

        row = int(coarse_row * stride)
        column = int(coarse_column * stride)
        radius = 2 * stride
        row_start = max(0, row - radius)
        row_stop = min(self.latitude.shape[0], row + radius + 1)
        column_start = max(0, column - radius)
        column_stop = min(self.latitude.shape[1], column + radius + 1)
        selection = np.s_[row_start:row_stop, column_start:column_stop]
        distance = self._distance(
            self.latitude[selection],
            self.longitude[selection],
            latitude,
            longitude,
        )
        distance[~self.valid[selection]] = np.inf
        local_row, local_column = np.unravel_index(
            int(np.argmin(distance)), distance.shape
        )
        return row_start + int(local_row), column_start + int(local_column)

    @staticmethod
    def _distance(
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        latitude: float,
        longitude: float,
    ) -> np.ndarray:
        longitude_delta = np.abs(longitudes - longitude)
        longitude_delta = np.minimum(longitude_delta, 360.0 - longitude_delta)
        return np.abs(latitudes - latitude) + longitude_delta


class ABIArchive:
    """Locate complete local ABI scans and crop channel-aligned chips."""

    CHANNELS = tuple(range(1, 17))

    def __init__(
        self,
        root: Path,
        geometry: ABIGeometry,
        satellite: SatelliteSpec,
        max_delta_minutes: float = 8.0,
    ):
        self.root = Path(root)
        self.geometry = geometry
        self.satellite = satellite
        self.max_delta = timedelta(minutes=max_delta_minutes)
        self._scan_cache: dict[
            tuple[int, int, int], dict[datetime, dict[int, Path]]
        ] = {}
        self._unreadable_scans: set[datetime] = set()

    def scans_for_hour(self, when: datetime) -> dict[datetime, dict[int, Path]]:
        key = (when.year, int(when.strftime("%j")), when.hour)
        if key in self._scan_cache:
            return self._scan_cache[key]

        directory = self.root / str(key[0]) / f"{key[1]:03d}" / f"{key[2]:02d}"
        scans: dict[datetime, dict[int, Path]] = defaultdict(dict)
        if directory.is_dir():
            for path in directory.iterdir():
                info = ABIFileInfo.from_path(path)
                if info is None or info.platform != self.satellite.platform_code:
                    continue
                scans[info.timestamp][info.channel] = path
        result = dict(scans)
        self._scan_cache[key] = result
        return result

    def nearest_scan(self, requested: datetime) -> tuple[datetime, dict[int, Path]]:
        return self.candidate_scans(requested)[0]

    def candidate_scans(
        self, requested: datetime
    ) -> list[tuple[datetime, dict[int, Path]]]:
        """Return complete readable-candidate scans ordered by time distance."""
        candidates: dict[datetime, dict[int, Path]] = {}
        for hour_delta in (-1, 0, 1):
            candidates.update(
                self.scans_for_hour(requested + timedelta(hours=hour_delta))
            )
        required = set(self.CHANNELS)
        complete = {
            timestamp: files
            for timestamp, files in candidates.items()
            if required.issubset(files)
            and timestamp not in self._unreadable_scans
        }
        if not complete:
            raise FileNotFoundError(
                f"No complete 16-channel {self.satellite.name} ABI scan near "
                f"{requested.isoformat()}"
            )
        ordered = sorted(complete, key=lambda value: abs(value - requested))
        within_tolerance = [
            timestamp
            for timestamp in ordered
            if abs(timestamp - requested) <= self.max_delta
        ]
        if not within_tolerance:
            scan_time = ordered[0]
            difference = abs(scan_time - requested)
            raise FileNotFoundError(
                f"Nearest ABI scan is {difference.total_seconds() / 60:.1f} minutes "
                f"from {requested.isoformat()} (limit "
                f"{self.max_delta.total_seconds() / 60:g})"
            )
        return [
            (timestamp, complete[timestamp])
            for timestamp in within_tolerance
        ]

    def crop(
        self, requested: datetime, row: int, column: int, size: int
    ) -> tuple[np.ndarray, datetime]:
        failures = []
        while True:
            try:
                candidates = self.candidate_scans(requested)
            except FileNotFoundError:
                if failures:
                    break
                raise
            scan_time, paths = candidates[0]
            try:
                channels = [
                    self._crop_channel(paths[channel], row, column, size)
                    for channel in self.CHANNELS
                ]
            except OSError as exc:
                self._unreadable_scans.add(scan_time)
                failures.append((scan_time, exc))
                LOG.warning(
                    "Quarantining unreadable ABI scan %s: %s",
                    scan_time.isoformat(),
                    exc,
                )
                continue
            return np.stack(channels, axis=-1), scan_time

        details = "; ".join(
            f"{timestamp.isoformat()}: {error}"
            for timestamp, error in failures
        )
        raise FileNotFoundError(
            f"No readable complete ABI scan near {requested.isoformat()}"
            + (f" ({details})" if details else "")
        )

    def _crop_channel(
        self, path: Path, row: int, column: int, size: int
    ) -> np.ndarray:
        nc = require_netcdf4()
        with nc.Dataset(path) as dataset:
            variable = dataset.variables["Rad"]
            scale = variable.shape[0] / self.geometry.latitude.shape[0]
            if (
                not np.isclose(scale, round(scale))
                and not np.isclose(1 / scale, round(1 / scale))
            ):
                raise ValueError(
                    f"Unsupported ABI channel resolution {variable.shape} in {path}"
                )
            native_row = int(round(row * scale))
            native_column = int(round(column * scale))
            native_half = max(1, int(round((size // 2) * scale)))
            row_slice = slice(native_row - native_half, native_row + native_half)
            column_slice = slice(
                native_column - native_half, native_column + native_half
            )
            if (
                row_slice.start < 0
                or column_slice.start < 0
                or row_slice.stop > variable.shape[0]
                or column_slice.stop > variable.shape[1]
            ):
                raise ValueError(
                    f"Chip centered at {(row, column)} extends outside ABI grid"
                )
            raw = variable[row_slice, column_slice]
            if np.ma.isMaskedArray(raw):
                raw = raw.filled(np.nan)
            chip = np.asarray(raw, dtype=np.float32)

        if chip.shape != (size, size):
            row_index = np.linspace(0, chip.shape[0] - 1, size).round().astype(int)
            column_index = (
                np.linspace(0, chip.shape[1] - 1, size).round().astype(int)
            )
            chip = chip[np.ix_(row_index, column_index)]
        return chip
