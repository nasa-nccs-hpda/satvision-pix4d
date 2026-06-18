"""Nearest-point sampling from local MERRA-2 NetCDF collections."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from satvision_pix4d.preprocessing.cloudsat_abi.utils import (
    as_datetime,
    normalize_longitude,
    require_netcdf4,
)


class MERRA2Reader:
    """Find same-date MERRA-2 collections and sample a space-time point."""

    def __init__(self, root: Path, variables: Sequence[str] = ()):
        self.root = Path(root)
        self.variables = tuple(variables)
        self._file_cache: dict[str, list[Path]] = {}

    def files_for_date(self, when: datetime) -> list[Path]:
        date_key = when.strftime("%Y%m%d")
        if date_key in self._file_cache:
            return self._file_cache[date_key]
        tokens = (date_key, when.strftime("%Y%j"))
        matches = sorted(
            path
            for path in self.root.rglob("*")
            if path.suffix.lower() in {".nc", ".nc4"}
            and any(token in path.name for token in tokens)
        )
        if not matches:
            raise FileNotFoundError(
                f"No MERRA-2 NetCDF file for {when.date()} below {self.root}"
            )
        self._file_cache[date_key] = matches
        return matches

    def sample(
        self, when: datetime, latitude: float, longitude: float
    ) -> tuple[dict[str, np.ndarray], list[Path]]:
        nc = require_netcdf4()
        sampled: dict[str, np.ndarray] = {}
        remaining = set(self.variables)
        sources: list[Path] = []
        for path in self.files_for_date(when):
            with nc.Dataset(path) as dataset:
                lat_var = self._coordinate(dataset, ("lat", "latitude"))
                lon_var = self._coordinate(dataset, ("lon", "longitude"))
                time_var = self._coordinate(dataset, ("time",))
                lat_values = np.asarray(lat_var[:])
                lon_values = np.asarray(lon_var[:])
                lat_index = int(np.argmin(np.abs(lat_values - latitude)))
                use_360 = bool(np.nanmax(lon_values) > 180)
                target_lon = float(normalize_longitude(longitude, use_360))
                lon_delta = np.abs(
                    normalize_longitude(lon_values, use_360) - target_lon
                )
                lon_index = int(np.argmin(np.minimum(lon_delta, 360 - lon_delta)))
                dates = nc.num2date(
                    time_var[:],
                    time_var.units,
                    getattr(time_var, "calendar", "standard"),
                )
                time_index = min(
                    range(len(dates)),
                    key=lambda index: abs(as_datetime(dates[index]) - when),
                )
                available = [
                    name
                    for name, variable in dataset.variables.items()
                    if name not in {lat_var.name, lon_var.name, time_var.name}
                    and variable.ndim >= 3
                    and (not self.variables or name in remaining)
                ]
                for name in available:
                    sampled[name] = self._sample_variable(
                        dataset.variables[name], time_index, lat_index, lon_index
                    )
                    remaining.discard(name)
                if available:
                    sources.append(path)
                sampled.update(
                    latitude=np.asarray(lat_values[lat_index]),
                    longitude=np.asarray(lon_values[lon_index]),
                    time=np.asarray(as_datetime(dates[time_index]).isoformat()),
                )
            if self.variables and not remaining:
                break
        if remaining:
            raise KeyError(
                f"MERRA-2 variable(s) {sorted(remaining)} not found in same-date "
                f"files below {self.root}"
            )
        return sampled, sources

    @staticmethod
    def _coordinate(dataset: Any, names: Sequence[str]):
        variables = {name.lower(): name for name in dataset.variables}
        for candidate in names:
            if candidate in variables:
                return dataset.variables[variables[candidate]]
        raise KeyError(f"None of coordinate variables {names} found")

    @staticmethod
    def _sample_variable(
        variable: Any, time_index: int, lat_index: int, lon_index: int
    ) -> np.ndarray:
        index: list[int | slice] = []
        for dimension in variable.dimensions:
            lower = dimension.lower()
            if lower == "time":
                index.append(time_index)
            elif lower in {"lat", "latitude"}:
                index.append(lat_index)
            elif lower in {"lon", "longitude"}:
                index.append(lon_index)
            else:
                index.append(slice(None))
        value = variable[tuple(index)]
        if np.ma.isMaskedArray(value):
            value = value.filled(np.nan)
        return np.asarray(value)
