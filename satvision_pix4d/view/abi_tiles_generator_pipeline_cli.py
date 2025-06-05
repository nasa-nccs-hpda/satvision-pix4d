import sys
import time
import logging
import argparse
from multiprocessing import cpu_count
from satvision_pix4d.pipelines.abi_tiles_generator_pipeline import \
    ABITileExtractor


# -----------------------------------------------------------------------------
# main
#
# python tiles_generator_pipeline_cli.py
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Use this application to generate tiles from raster data.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-r',
                        '--stratification',
                        type=str,
                        required=False,
                        default='random',
                        choices=['convection'],
                        dest='stratification_strategy',
                        help='Select the stratification strategy')

    parser.add_argument('-o',
                        '--output-dir',
                        type=str,
                        default='./abi_tiles',
                        required=False,
                        dest='output_dir',
                        help='Path to output directory')

    """
    parser.add_argument('-r',
                        '--input-regex',
                        type=str,
                        required=True,
                        default=None,
                        dest='input_regex',
                        help='Input regex to select files')

    
    parser.add_argument('-ts',
                        '--tile-size',
                        type=int,
                        required=False,
                        default=64,
                        dest='tile_size',
                        help='Integer to represent square tile size')

    parser.add_argument('-nt',
                        '--n-tiles',
                        type=int,
                        required=False,
                        default=2000,
                        dest='n_tiles',
                        help='Integer with number of tiles per image')

    parser.add_argument('-s',
                        '--stride',
                        type=int,
                        required=False,
                        default=None,
                        dest='stride',
                        help='Integer with stride for overlapping tiles')

    parser.add_argument('-e',
                        '--output-extension',
                        type=str,
                        required=False,
                        default='npy',
                        choices=['tif', 'npy'],
                        dest='output_extension',
                        help='String with output extension (tif, npy)')

    parser.add_argument('-nw',
                        '--n-workers',
                        type=int,
                        required=False,
                        default=cpu_count(),
                        dest='n_workers',
                        help='Integer with number of simultaneous workers')


    """

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

    # Set pipeline
    pipeline = ABITileExtractor(
        stratification_strategy=args.stratification_strategy,
        output_dir=args.output_dir,
        #tile_size=args.tile_size,
        #stride=args.stride,
        #num_tiles_per_image=args.n_tiles,
        #output_extension=args.output_extension,
        #num_workers=args.n_workers
    )

    # Process images
    pipeline.gen_tiles()

    logging.info(f'Took {(time.time()-timer)/60.0:.2f} min.')

    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
