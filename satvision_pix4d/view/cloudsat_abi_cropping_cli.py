"""Command-line interface for CloudSat and ABI collocation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from satvision_pix4d.preprocessing.cloudsat_abi import (
    CloudSatABICollocationPipeline,
    CropConfig,
    get_satellite,
)
from satvision_pix4d.preprocessing.cloudsat_abi.config import SATELLITES


LOG = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Crop multitemporal GOES ABI chips colocated with a CloudSat "
            "transect."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""example:
  python scripts/crop_abi_multitemporal_rewriting.py \\
    --abi-root /data/GOES-16-ABI-L1B-FULLD \\
    --cloudsat-root /data/cloudsat \\
    --latlon-path /data/ABI_EAST_GEO_TOPO_LOMSK.nc \\
    --output-dir /data/chips --year 2019 --day-start 335 --day-end 336 \\
    --transect -30 30 --satellite goes16 --metadata cloudsat merra2 \\
    --merra2-root /data/MERRA2 --merra2-variables T QV U V
""",
    )
    parser.add_argument(
        "--abi-root", type=Path, required=True,
        help="ABI root organized as YYYY/DDD/HH",
    )
    parser.add_argument(
        "--cloudsat-root", type=Path, required=True,
        help="Root containing the 2B-CLDCLASS-LIDAR product directory",
    )
    parser.add_argument(
        "--cloudsat-index-root", type=Path,
        help="Direct 2B-CLDCLASS-LIDAR path; defaults below --cloudsat-root",
    )
    parser.add_argument(
        "--latlon-path", type=Path, required=True,
        help="ABI East or West Latitude/Longitude NetCDF grid",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--day-start", type=int, default=1)
    parser.add_argument("--day-end", type=int)
    parser.add_argument("--orbit", help="Process only this five-digit CloudSat orbit")
    parser.add_argument(
        "--transect", type=float, nargs=2,
        metavar=("LAT_MIN", "LAT_MAX"), required=True,
        help="Inclusive CloudSat latitude bounds",
    )
    parser.add_argument(
        "--satellite", choices=sorted(SATELLITES), required=True,
        help="Physical GOES spacecraft; East/West is inferred and recorded",
    )
    parser.add_argument(
        "--offsets", type=int, nargs="+", default=[-40, -20, 0, 20, 40]
    )
    parser.add_argument("--chip-size", type=int, default=128)
    parser.add_argument("--profile-stride", type=int, default=45)
    parser.add_argument("--profiles-per-chip", type=int, default=91)
    parser.add_argument(
        "--metadata", nargs="*", choices=("cloudsat", "merra2"),
        default=["cloudsat"],
        help="Metadata groups to store; pass with no values for none",
    )
    parser.add_argument("--merra2-root", type=Path)
    parser.add_argument(
        "--merra2-variables", nargs="*", default=[],
        help="MERRA-2 variables; empty means all gridded variables",
    )
    parser.add_argument("--max-scan-delta-minutes", type=float, default=8.0)
    parser.add_argument("--allow-missing-timesteps", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-chips", type=int,
        help="Stop after this many new outputs (useful for validation)",
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> CropConfig:
    index_root = (
        args.cloudsat_index_root
        or args.cloudsat_root / "2B-CLDCLASS-LIDAR"
    )
    return CropConfig(
        abi_root=args.abi_root,
        cloudsat_root=args.cloudsat_root,
        cloudsat_index_root=index_root,
        latlon_path=args.latlon_path,
        output_dir=args.output_dir,
        year=args.year,
        satellite=get_satellite(args.satellite),
        day_start=args.day_start,
        day_end=args.day_end,
        orbit=args.orbit,
        transect=tuple(args.transect),
        offsets=tuple(args.offsets),
        chip_size=args.chip_size,
        profile_stride=args.profile_stride,
        profiles_per_chip=args.profiles_per_chip,
        metadata=frozenset(args.metadata),
        merra2_root=args.merra2_root,
        merra2_variables=tuple(args.merra2_variables),
        max_scan_delta_minutes=args.max_scan_delta_minutes,
        allow_missing_timesteps=args.allow_missing_timesteps,
        overwrite=args.overwrite,
        max_chips=args.max_chips,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        config = config_from_args(args)
        written = CloudSatABICollocationPipeline(config).run()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOG.error("%s", exc)
        return 1
    LOG.info("Finished: %d chip(s) written", written)
    return 0
