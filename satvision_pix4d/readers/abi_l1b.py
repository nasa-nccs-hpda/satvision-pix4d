"""Advance Baseline Imager reader for the Level 1b format.

The files read by this reader are described in the official PUG document:

    https://www.goes-r.gov/users/docs/PUG-L1b-vol3.pdf

"""
import logging

logger = logging.getLogger(__name__)


class ABI_L1B_Reader(object):
    """File reader for individual ABI L1B NetCDF4 files."""

    def __init__(self, filename: str = None):
        print('Reading ABI data')

    def download(
            self,
            to_disk: bool = False,
            output_dir: str = None
        ):
        print('Load from AWS')
        return
