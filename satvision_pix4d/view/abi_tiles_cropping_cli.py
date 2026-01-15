import os
import sys
import time
import logging
import argparse
import warnings
from multiprocessing import cpu_count
from rasterio.errors import NotGeoreferencedWarning
from satvision_pix4d.pipelines.abi_tiles_cropping import ABICloudSatCropping


# -----------------------------------------------------------------------------
# main
#
# python tiles_generator_pipeline_cli.py
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Use this application to generate tiles from raster data.'
    parser = argparse.ArgumentParser(description=desc)

    # translation = [1, 2, 0, 4, 5, 6, 3, 8, 9, 10, 11, 13, 14, 15]

    # LATLONDATA
    parser.add_argument(
        '-llp',
        '--lat-lon-path',
        type=str,
        default="/explore/nobackup/projects/pix4dcloud/jgong/ABI_EAST_GEO_TOPO_LOMSK.nc",
        required=False,
        dest='latlon_path',
        help='Path to lat lon NC file.')

    # CLOUDSATPATH
    parser.add_argument(
        '-c',
        '--cloudsat-path',
        type=str,
        default="/explore/nobackup/projects/pix4dcloud/szhang16/cloudsat",
        required=False,
        dest='cloudsat_path',
        help='Path to cloudsat data.')

    # ROOT_DIR
    parser.add_argument(
        '-r',
        '--root-dir',
        type=str,
        default="/explore/nobackup/projects/pix4dcloud/szhang16/cloudsat/2B-CLDCLASS-LIDAR",
        required=False,
        dest='root_dir',
        help='Path to root dir.')

    # ABIDATA
    parser.add_argument(
        '-a',
        '--abi-data',
        type=str,
        default="/css/geostationary/BackStage/GOES-16-ABI-L1B-FULLD",
        required=False,
        dest='abidata_path',
        help='Path to abi data.')

    parser.add_argument(
        '-y'
        '--year',
        type=str,
        required=False,
        default="2019",
        dest='year',
        help='Year to process'
    )

    # SAVEDIR
    parser.add_argument(
        '-o',
        '--output-dir',
        type=str,
        default='./abi_tiles',
        required=False,
        dest='output_dir',
        help='Path to output directory')

    parser.add_argument(
        '-d',
        '--day-skip',
        type=int,
        required=False,
        default=0,
        dest='dayskip',
        help='Integer for day to skip'
    )

    args = parser.parse_args()

    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)  # set stdout handler
    ch.setLevel(logging.INFO)

    # set formatter and handlers
    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Setup timer to monitor script execution time
    timer = time.time()

    logging.info("CloudSat + ABI Tile Croppping")

    pipeline = ABICloudSatCropping(
        latlon_data=args.latlon_path,
        root_dir=args.root_dir,
        cloudsat_path=args.cloudsat_path,
        abidata_path=args.abidata_path,
        output_dir=args.output_dir,
        year=args.year,
        dayskip=args.dayskip
    )

    pipeline.gen_tiles()

    logging.info(f'Took {(time.time()-timer)/60.0:.2f} min.')
    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
