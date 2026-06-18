"""Orchestration for CloudSat and multitemporal ABI collocation."""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np

from satvision_pix4d.preprocessing.cloudsat_abi.abi import ABIArchive, ABIGeometry
from satvision_pix4d.preprocessing.cloudsat_abi.cloudsat import (
    CloudSatOrbitFile,
    CloudSatReader,
    CloudSatTransect,
)
from satvision_pix4d.preprocessing.cloudsat_abi.config import CropConfig
from satvision_pix4d.preprocessing.cloudsat_abi.merra2 import MERRA2Reader
from satvision_pix4d.preprocessing.cloudsat_abi.utils import datetime_from_year_doy
from satvision_pix4d.preprocessing.cloudsat_abi.writer import (
    CollocatedChip,
    NPZChipWriter,
)


LOG = logging.getLogger(__name__)


class CloudSatABICollocationPipeline:
    """Build and write ABI time-series chips along CloudSat transects."""

    def __init__(
        self,
        config: CropConfig,
        abi_archive: ABIArchive | None = None,
        cloudsat_reader: CloudSatReader | None = None,
        merra2_reader: MERRA2Reader | None = None,
        writer: NPZChipWriter | None = None,
    ):
        self.config = config
        if abi_archive is None:
            geometry = ABIGeometry(config.latlon_path)
            abi_archive = ABIArchive(
                config.abi_root,
                geometry,
                config.satellite,
                config.max_scan_delta_minutes,
            )
        self.abi = abi_archive
        self.cloudsat = cloudsat_reader or CloudSatReader(
            config.cloudsat_root, config.cloudsat_index_root
        )
        self.merra2 = merra2_reader
        if "merra2" in config.metadata and self.merra2 is None:
            assert config.merra2_root is not None
            self.merra2 = MERRA2Reader(
                config.merra2_root, config.merra2_variables
            )
        self.writer = writer or NPZChipWriter(
            config.output_dir, config.overwrite
        )

    def run(self) -> int:
        written = 0
        for orbit_file in self.cloudsat.discover_orbits(
            self.config.year,
            self.config.day_start,
            self.config.day_end,
            self.config.orbit,
        ):
            LOG.info(
                "Processing CloudSat orbit %s on %d-%03d",
                orbit_file.orbit,
                self.config.year,
                orbit_file.day,
            )
            transect = self.cloudsat.read(
                orbit_file.path, self.config.transect
            )
            if len(transect) < self.config.profiles_per_chip:
                LOG.warning(
                    "Skipping orbit %s: only %d profiles in transect",
                    orbit_file.orbit,
                    len(transect),
                )
                continue
            for center in self._profile_centers(transect):
                if (
                    self.config.max_chips is not None
                    and written >= self.config.max_chips
                ):
                    return written
                try:
                    sample = self.build_sample(orbit_file, transect, center)
                    output, created = self.writer.write(sample)
                except (
                    FileNotFoundError,
                    ValueError,
                    IndexError,
                    KeyError,
                ) as exc:
                    LOG.warning(
                        "Skipping orbit %s profile %d: %s",
                        orbit_file.orbit,
                        center,
                        exc,
                    )
                    continue
                written += int(created)
                LOG.info("%s %s", "Saved" if created else "Exists", output)
        return written

    def _profile_centers(self, transect: CloudSatTransect) -> range:
        margin = self.config.profiles_per_chip // 2
        return range(
            margin,
            len(transect) - margin,
            self.config.profile_stride,
        )

    def build_sample(
        self,
        orbit_file: CloudSatOrbitFile,
        transect: CloudSatTransect,
        center: int,
    ) -> CollocatedChip:
        center_time = datetime_from_year_doy(
            self.config.year, orbit_file.day, transect.utc_hour[center]
        )
        center_latitude = float(transect.latitude[center])
        center_longitude = float(transect.longitude[center])
        row, column = self.abi.geometry.nearest(
            center_latitude, center_longitude
        )
        chips, scan_times, valid = [], [], []
        for offset in self.config.offsets:
            requested = center_time + timedelta(minutes=offset)
            try:
                chip, scan_time = self.abi.crop(
                    requested, row, column, self.config.chip_size
                )
                valid.append(1)
            except (FileNotFoundError, ValueError) as exc:
                if not self.config.allow_missing_timesteps:
                    raise
                LOG.warning("Missing timestep %s: %s", requested.isoformat(), exc)
                chip = np.full(
                    (
                        self.config.chip_size,
                        self.config.chip_size,
                        16,
                    ),
                    np.nan,
                    dtype=np.float32,
                )
                scan_time = None
                valid.append(0)
            chips.append(chip)
            scan_times.append(scan_time.isoformat() if scan_time else "")

        timestamp = center_time.strftime("%Y%m%dT%H%M%SZ")
        filename = (
            f"{self.config.satellite.filename_token}_"
            f"{self.config.satellite.region}_abi_cloudsat_{timestamp}_"
            f"orbit{orbit_file.orbit}_r{row}_c{column}_p{center}.npz"
        )
        metadata = self._metadata(
            orbit_file,
            center,
            center_time.isoformat(),
            center_latitude,
            center_longitude,
            row,
            column,
        )
        auxiliary: dict[str, np.ndarray] = {}
        if "cloudsat" in self.config.metadata:
            auxiliary.update(
                transect.metadata_arrays(
                    center, self.config.profiles_per_chip
                )
            )
        if "merra2" in self.config.metadata:
            assert self.merra2 is not None
            values, sources = self.merra2.sample(
                center_time, center_latitude, center_longitude
            )
            auxiliary.update(
                {f"merra2_{name}": value for name, value in values.items()}
            )
            metadata["merra2_sources"] = [str(path) for path in sources]

        return CollocatedChip(
            filename=filename,
            chip=np.stack(chips).astype(np.float32),
            offsets_minutes=np.asarray(self.config.offsets, dtype=np.int32),
            valid_mask=np.asarray(valid, dtype=np.int8),
            scan_times=np.asarray(scan_times),
            metadata=metadata,
            auxiliary_arrays=auxiliary,
        )

    def _metadata(
        self,
        orbit_file: CloudSatOrbitFile,
        center: int,
        center_time: str,
        latitude: float,
        longitude: float,
        row: int,
        column: int,
    ) -> dict:
        return {
            "schema_version": 1,
            "satellite": self.config.satellite.name,
            "satellite_region": self.config.satellite.region,
            "cloudsat_orbit": orbit_file.orbit,
            "cloudsat_center_profile": center,
            "cloudsat_center_time": center_time,
            "center_latitude": latitude,
            "center_longitude": longitude,
            "abi_row": row,
            "abi_column": column,
            "chip_size": self.config.chip_size,
            "transect_latitude_min": self.config.transect[0],
            "transect_latitude_max": self.config.transect[1],
            "metadata_groups": sorted(self.config.metadata),
            "abi_root": str(self.config.abi_root),
            "abi_geometry_source": str(self.config.latlon_path),
            "cloudsat_source": str(orbit_file.path),
        }
