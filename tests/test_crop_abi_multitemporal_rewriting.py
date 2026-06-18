from datetime import datetime, timezone

import numpy as np
import pytest

from satvision_pix4d.preprocessing.cloudsat_abi.abi import (
    ABIArchive,
    ABIFileInfo,
    ABIGeometry,
)
from satvision_pix4d.preprocessing.cloudsat_abi.cloudsat import (
    CloudSatOrbitFile,
    CloudSatTransect,
)
from satvision_pix4d.preprocessing.cloudsat_abi.config import (
    CropConfig,
    get_satellite,
)
from satvision_pix4d.preprocessing.cloudsat_abi.pipeline import (
    CloudSatABICollocationPipeline,
)
from satvision_pix4d.preprocessing.cloudsat_abi.writer import NPZChipWriter
from satvision_pix4d.view.cloudsat_abi_cropping_cli import (
    build_parser,
    config_from_args,
)


def abi_name(channel, platform="G16", stamp="2019336235959"):
    return (
        f"OR_ABI-L1b-RadF-M6C{channel:02d}_{platform}_s{stamp}_"
        "e2019337000959_c.nc"
    )


def make_config(tmp_path, **overrides):
    values = {
        "abi_root": tmp_path / "abi",
        "cloudsat_root": tmp_path / "cloudsat",
        "cloudsat_index_root": tmp_path / "cloudsat" / "2B-CLDCLASS-LIDAR",
        "latlon_path": tmp_path / "grid.nc",
        "output_dir": tmp_path / "output",
        "year": 2019,
        "satellite": get_satellite("goes18"),
        "transect": (-30.0, 30.0),
        "offsets": (-20, 0, 20),
        "chip_size": 8,
        "profile_stride": 2,
        "profiles_per_chip": 5,
        "metadata": frozenset({"cloudsat"}),
    }
    values.update(overrides)
    return CropConfig(**values)


def make_transect(tmp_path, profiles=9):
    base = np.full((profiles, 3), -1.0)
    top = np.full((profiles, 3), -1.0)
    cloud_type = np.zeros((profiles, 3), dtype=np.int8)
    base[:, 0], top[:, 0], cloud_type[:, 0] = 2.0, 3.0, 4
    return CloudSatTransect(
        source=tmp_path / "orbit.hdf",
        latitude=np.linspace(-4, 4, profiles),
        longitude=np.linspace(-80, -72, profiles),
        utc_hour=np.full(profiles, 12.0),
        quality=np.zeros(profiles, dtype=np.int8),
        cloud_layer_base=base,
        cloud_layer_top=top,
        cloud_layer_type=cloud_type,
    )


def test_abi_file_info_reads_platform_channel_and_year_rollover():
    info = ABIFileInfo.from_path(abi_name(7))

    assert info is not None
    assert info.timestamp == datetime(
        2019, 12, 2, 23, 59, 59, tzinfo=timezone.utc
    )
    assert info.channel == 7
    assert info.platform == "G16"


def test_archive_selects_complete_requested_platform_scan_across_hour(tmp_path):
    directory = tmp_path / "2019" / "336" / "23"
    directory.mkdir(parents=True)
    for channel in range(1, 17):
        (directory / abi_name(channel)).touch()
        (directory / abi_name(channel, platform="G17")).touch()

    geometry = type("GeometryStub", (), {"latitude": np.zeros((8, 8))})()
    archive = ABIArchive(
        tmp_path, geometry, get_satellite("goes16"), max_delta_minutes=2
    )
    scan_time, paths = archive.nearest_scan(
        datetime(2019, 12, 3, 0, 0, 30, tzinfo=timezone.utc)
    )

    assert scan_time == datetime(
        2019, 12, 2, 23, 59, 59, tzinfo=timezone.utc
    )
    assert set(paths) == set(range(1, 17))
    assert all("_G16_" in path.name for path in paths.values())


def test_geometry_nearest_uses_local_refinement():
    geometry = ABIGeometry.__new__(ABIGeometry)
    rows, columns = np.mgrid[0:100, 0:100]
    geometry.latitude = rows.astype(np.float32)
    geometry.longitude = columns.astype(np.float32)
    geometry.valid = np.ones((100, 100), dtype=bool)
    geometry.use_360 = False
    geometry.lat_min = 0.0
    geometry.lat_max = 99.0
    geometry.coarse_target_size = 256

    assert geometry.nearest(52.1, 37.2) == (52, 37)


def test_cloudsat_transect_owns_window_and_cloud_mask_logic(tmp_path):
    transect = make_transect(tmp_path)

    arrays = transect.metadata_arrays(center=4, count=5)

    assert arrays["cloudsat_latitude"].shape == (5,)
    assert arrays["cloudsat_cloud_mask"].shape == (5, 40)
    assert arrays["cloudsat_cloud_mask"][0, 4:7].tolist() == [4, 4, 4]
    with pytest.raises(IndexError):
        transect.profile_window(center=1, count=5)


def test_crop_config_validates_merra_and_normalizes_transect(tmp_path):
    config = make_config(tmp_path, transect=(30.0, -30.0))
    assert config.transect == (-30.0, 30.0)

    with pytest.raises(ValueError, match="merra2_root"):
        make_config(tmp_path, metadata=frozenset({"merra2"}))


def test_cli_builds_typed_satellite_configuration(tmp_path):
    args = build_parser().parse_args([
        "--abi-root", str(tmp_path / "abi"),
        "--cloudsat-root", str(tmp_path / "cloudsat"),
        "--latlon-path", str(tmp_path / "west.nc"),
        "--output-dir", str(tmp_path / "output"),
        "--year", "2019",
        "--transect", "-30", "30",
        "--satellite", "goes18",
        "--metadata",
    ])

    config = config_from_args(args)

    assert config.satellite.name == "GOES-18"
    assert config.satellite.region == "west"
    assert config.metadata == frozenset()


@pytest.mark.parametrize(
    ("satellite", "filename"),
    [
        ("goes16", "ABI_EAST_GEO_TOPO_LOMSK.nc"),
        ("goes18", "ABI_WEST_GEO_TOPO_LOMSK.nc"),
        ("goes19", "ABI_EAST_GEO_TOPO_LOMSK.nc"),
    ],
)
def test_cli_infers_geometry_file_from_satellite(tmp_path, satellite, filename):
    parser = build_parser()
    args = parser.parse_args([
        "--abi-root", str(tmp_path / "abi"),
        "--cloudsat-root", str(tmp_path / "cloudsat"),
        "--latlon-dir", str(tmp_path / "geometry"),
        "--output-dir", str(tmp_path / "output"),
        "--year", "2019",
        "--transect", "-30", "30",
        "--satellite", satellite,
    ])

    config = config_from_args(args)

    assert config.latlon_path == tmp_path / "geometry" / filename


def test_explicit_geometry_file_overrides_directory_inference(tmp_path):
    parser = build_parser()
    override = tmp_path / "custom-grid.nc"
    args = parser.parse_args([
        "--abi-root", str(tmp_path / "abi"),
        "--cloudsat-root", str(tmp_path / "cloudsat"),
        "--latlon-dir", str(tmp_path / "geometry"),
        "--latlon-path", str(override),
        "--output-dir", str(tmp_path / "output"),
        "--year", "2019",
        "--transect", "-30", "30",
        "--satellite", "goes18",
    ])

    assert config_from_args(args).latlon_path == override


def test_pipeline_components_are_injectable_and_preserve_output_schema(tmp_path):
    transect = make_transect(tmp_path)
    orbit_file = CloudSatOrbitFile(336, "72433", transect.source)

    class FakeGeometry:
        @staticmethod
        def nearest(latitude, longitude):
            return 10, 20

    class FakeABI:
        geometry = FakeGeometry()

        @staticmethod
        def crop(requested, row, column, size):
            return np.zeros((size, size, 16), dtype=np.float32), requested

    class FakeCloudSatReader:
        @staticmethod
        def discover_orbits(year, day_start, day_end, orbit):
            return [orbit_file]

        @staticmethod
        def read(path, latitude_bounds):
            return transect

    config = make_config(tmp_path, max_chips=1)
    pipeline = CloudSatABICollocationPipeline(
        config,
        abi_archive=FakeABI(),
        cloudsat_reader=FakeCloudSatReader(),
        writer=NPZChipWriter(config.output_dir),
    )

    assert pipeline.run() == 1
    outputs = list(config.output_dir.glob("*.npz"))
    assert len(outputs) == 1
    assert outputs[0].name.startswith("GOES18_west_")
    with np.load(outputs[0]) as data:
        assert data["chip"].shape == (3, 8, 8, 16)
        assert data["cloudsat_cloud_mask"].shape == (5, 40)
        metadata = str(data["metadata_json"])
        assert '"satellite":"GOES-18"' in metadata
        assert '"satellite_region":"west"' in metadata
