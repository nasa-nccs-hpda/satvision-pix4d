import sys
import time
import logging
import argparse
import warnings
from multiprocessing import cpu_count
from rasterio.errors import NotGeoreferencedWarning

# Suppress the NotGeoreferencedWarning only
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

from satvision_pix4d.pipelines.abi_tiles_generator_convection_pipeline import \
    ABIConvectionTileExtractor

from satvision_pix4d.pipelines.abi_tiles_generator_random_pipeline import \
    ABIRandomTileExtractor


# -----------------------------------------------------------------------------
# main
#
# python tiles_generator_pipeline_cli.py
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Use this application to generate tiles from raster data.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '-r',
        '--stratification',
        type=str,
        required=False,
        default='random',
        choices=[
            'random',
            'convection',
            'clouds',
            'landcover'
        ],
        dest='stratification_strategy',
        help='Select the stratification strategy')

    parser.add_argument(
        '-o',
        '--output-dir',
        type=str,
        default='./abi_tiles',
        required=False,
        dest='output_dir',
        help='Path to output directory')

    parser.add_argument(
        '-cr',
        '--convection-regex',
        type=str,
        required=False,
        default=None,
        dest='convection_regex',
        help='Glob pattern or directory containing convection mask files'
    )

    parser.add_argument(
        '-ts',
        '--tile-size',
        type=int,
        required=False,
        default=512,
        dest='tile_size',
        help='Integer to represent square tile size'
    )

    parser.add_argument(
        "-c",
        "--channels",
        type=int,
        nargs="+",
        default=list(range(1, 17)),
        dest="channels",
        help="List of ABI channel numbers to use (1–16). Example: -c 1 2 3 7"
    )

    parser.add_argument(
        '--local-data-dir',
        type=str,
        required=False,
        default=None,
        dest='local_data_dir',
        help='Path to directory where data might reside'
    )

    parser.add_argument(
        '-s',
        '--satellite',
        type=str,
        required=False,
        default='goes16',
        choices=['goes16', 'goes17', 'goes18'],
        dest='satellite',
        help='String with satellite to download data for')

    parser.add_argument(
        '-p',
        '--product',
        type=str,
        required=False,
        default='ABI-L1b-RadF',
        choices=['ABI-L1b-RadF'], # support just one for now
        dest='product',
        help='String with product to download data for')

    parser.add_argument(
        '-d',
        '--domain',
        type=str,
        required=False,
        default='F',
        choices=['F'], # support just one for now
        dest='domain',
        help='String with product to download data for')

    parser.add_argument(
        '--download',
        action='store_true',
        default=False,
        help='Download ABI data from AWS if local files are not found (default: False)'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        default=False,
        help='Overwrite any existing downloaded files (default: False)'
    )

    parser.add_argument('--num-tiles', type=int, default=5, help='Number of random tiles to generate')
    parser.add_argument('--start-dt', type=str, default="2019-01-01T00:00:00", help='Random sampling start datetime (ISO)')
    parser.add_argument('--end-dt', type=str, default="2019-01-10T23:59:59", help='Random sampling end datetime (ISO)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    args = parser.parse_args()

    # 🚨 Validate that --convection-regex is provided
    # if stratification is convection
    # add clouds and landcover validation here
    if args.stratification_strategy == 'convection' \
            and not args.convection_regex:
        parser.error(
            "The --convection-regex argument is required"
            " when stratification is 'convection'.")

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

    logging.info(
        f"Generating tiles via '{args.stratification_strategy}' strategy.")

    if args.stratification_strategy == 'random':
        pipeline = ABIRandomTileExtractor(
            output_dir=args.output_dir,
            tile_size=args.tile_size,
            channels=args.channels,
            satellite=args.satellite,
            product=args.product,
            domain=args.domain,
            download=args.download,
            overwrite=args.overwrite,
            num_tiles=args.num_tiles,
            start_dt=args.start_dt,
            end_dt=args.end_dt,
            seed=args.seed,
        )

    elif args.stratification_strategy == 'convection':
        pipeline = ABIConvectionTileExtractor(
            output_dir=args.output_dir,
            convection_regex=args.convection_regex,
            tile_size=args.tile_size,
            channels=args.channels,
            local_data_dir=args.local_data_dir,
            satellite=args.satellite,
            product=args.product,
            domain=args.domain,
            download=args.download,
            overwrite=args.overwrite,
        )

    pipeline.gen_tiles()

    logging.info(f'Took {(time.time()-timer)/60.0:.2f} min.')
    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
