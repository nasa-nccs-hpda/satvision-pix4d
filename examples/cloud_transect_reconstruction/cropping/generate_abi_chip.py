#!/usr/bin/env python
"""
Generate ABI Chips from Lat/Lon and UTC Time

Simple script to extract GOES-16 ABI satellite image chips given:
- Center point coordinates (latitude, longitude)
- UTC timestamp of ABI full disk image
- Desired chip size

Based on the CloudSat-ABI matching workflow from cropChips.py

Usage:
    from generate_abi_chip import generate_abi_chip
    chip = generate_abi_chip(lat=35.5, lon=-95.0, utc_time=datetime(2019, 9, 18, 20, 0))

Author: Based on cropChips.py workflow
Version: 1.0
"""

import numpy as np
import netCDF4 as nc
import os
import argparse
from datetime import datetime
from typing import Dict, Tuple, Union, Optional

# ============================================================================
# CONFIGURATION (modify these paths for your system)
# ============================================================================

# Path to pre-computed ABI lat/lon grid
LATLONDATA = "/explore/nobackup/projects/pix4dcloud/jgong/ABI_EAST_GEO_TOPO_LOMSK.nc"

# Path to ABI L1b Full Disk data
ABIDATA_ROOT = "/css/geostationary/NonOptimized/L1/GOES-17-ABI-L1B-FULLD/"

# Default output directory for generated chips
OUTPUT_ROOT = "/explore/nobackup/projects/pix4dcloud/jli30/abiChips/exp/amv_p128"

# Grid parameters
BOUND_SIZE = 1600  # Safety margin from disk edge (pixels)
LENGTH = 10848     # Full disk size (pixels)

# Default chip size
DEFAULT_CHIP_HALF_SIZE = 64  # Creates 128×128 chips

# ============================================================================
# Global variables (loaded once)
# ============================================================================

_geolocation_loaded = False
abiLong = None
abiLat = None
latSlice = None
longSlice = None
longMin = None
longMax = None
latMin = None
latMax = None


def load_geolocation_grid(latlon_file: str = LATLONDATA) -> None:
    """
    Load ABI geolocation grid into global variables.
    Called automatically on first use.
    
    Args:
        latlon_file: Path to ABI lat/lon grid NetCDF file
    
    Raises:
        FileNotFoundError: If geolocation file not found
    """
    global _geolocation_loaded, abiLong, abiLat, latSlice, longSlice
    global longMin, longMax, latMin, latMax
    
    if _geolocation_loaded:
        return
    
    print(f"Loading ABI geolocation grid from: {latlon_file}")
    
    try:
        f = nc.Dataset(latlon_file)
        abiLong = f['Longitude'][:]  # (10848, 10848)
        abiLat = f['Latitude'][:]    # (10848, 10848)
        f.close()
        
        # Create bounded region (remove edge artifacts)
        abiLongB = abiLong[BOUND_SIZE:LENGTH-BOUND_SIZE, BOUND_SIZE:LENGTH-BOUND_SIZE]
        abiLatB = abiLat[BOUND_SIZE:LENGTH-BOUND_SIZE, BOUND_SIZE:LENGTH-BOUND_SIZE]
        
        # Clean invalid values
        abiLongB[abiLongB == -999] = 10
        abiLatB[abiLatB == -999] = 10
        abiLongB[abiLongB < 0] += 360
        
        # Determine valid bounds
        longMin = float(abiLongB.min())
        longMax = float(abiLongB.max())
        latMin = float(abiLatB.min())
        latMax = float(abiLatB.max())
        
        # Create 1D slices for quick lookup
        latSlice = abiLat[:, 5424][18:-18][::-1]  # Center column, trimmed, reversed
        longSlice = abiLong[5424, :][18:-18]      # Center row, trimmed
        
        _geolocation_loaded = True
        
        print(f"✅ Grid loaded: {abiLat.shape}")
        print(f"   Valid coverage: Lat [{latMin:.2f}°, {latMax:.2f}°], "
              f"Lon [{longMin:.2f}°, {longMax:.2f}°]")
        
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Geolocation file not found: {latlon_file}\n"
            f"Please update LATLONDATA in the script configuration."
        )
    except Exception as e:
        raise RuntimeError(f"Error loading geolocation grid: {e}")


def parse_utc_to_abi_time(utc_datetime: datetime) -> Tuple[str, str, str, str]:
    """
    Convert Python datetime to ABI file path components.
    
    Args:
        utc_datetime: datetime object in UTC
    
    Returns:
        Tuple of (year, day_of_year, hour, minute_str)
    
    Example:
        >>> parse_utc_to_abi_time(datetime(2019, 9, 18, 20, 23, 0))
        ('2019', '261', '20', '20')
    """
    year = utc_datetime.strftime('%Y')
    doy = utc_datetime.strftime('%j')  # Day of year (001-366)
    hour = utc_datetime.strftime('%H')
    
    # Round to nearest 10-minute interval
    minute = utc_datetime.minute
    rounded_minute = int(np.round(minute / 10) * 10)
    
    # Handle hour rollover
    if rounded_minute == 60:
        from datetime import timedelta
        utc_datetime = utc_datetime + timedelta(hours=1)
        utc_datetime = utc_datetime.replace(minute=0)
        hour = utc_datetime.strftime('%H')
        rounded_minute = 0
    
    minute_str = f"{rounded_minute:02d}"
    
    return year, doy, hour, minute_str


def find_nearest_abi_pixel(
    lat: float,
    lon: float,
    validate_chip_region: bool = True,
    verbose: bool = False,
) -> Dict[str, object]:
    """
    Find the nearest ABI pixel to a lat/lon coordinate.

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees (0-360 or -180 to 180)
        validate_chip_region: If True, enforce the BOUND_SIZE safe region check
        verbose: Print boundary-search warning if fallback is needed

    Returns:
        Dict with keys:
        {'coords', 'matched_lat', 'matched_lon', 'distance_km', 'lon_normalized'}
    """
    global _geolocation_loaded

    if not _geolocation_loaded:
        load_geolocation_grid()

    lon_normalized = lon + 360 if lon < 0 else lon

    # Validate requested coordinate against ABI coverage.
    if lat < latMin or lat > latMax:
        raise ValueError(
            f"Latitude {lat:.2f}° outside valid GOES-16 coverage "
            f"[{latMin:.2f}°, {latMax:.2f}°]"
        )
    if lon_normalized < longMin or lon_normalized > longMax:
        raise ValueError(
            f"Longitude {lon_normalized:.2f}° outside valid GOES-16 coverage "
            f"[{longMin:.2f}°, {longMax:.2f}°]"
        )

    AREA_SIZE = 1000
    lati = len(latSlice) - np.searchsorted(latSlice, lat) + 17
    loni = np.searchsorted(longSlice, lon_normalized) + 18

    # Fine search in local window.
    lat_region = abiLat[lati-AREA_SIZE:lati+AREA_SIZE, loni-AREA_SIZE:loni+AREA_SIZE]
    lon_region = abiLong[lati-AREA_SIZE:lati+AREA_SIZE, loni-AREA_SIZE:loni+AREA_SIZE]
    distances = np.abs(lat_region - lat) + np.abs(lon_region - lon_normalized)
    coords = np.array(np.unravel_index(distances.argmin(), distances.shape), dtype=int)

    # Expand to full-disk search if the local minimum hits the search-window boundary.
    if coords[0] == 0 or coords[1] == 0 or \
       coords[1] == 2*AREA_SIZE-1 or coords[0] == 2*AREA_SIZE-1:
        if verbose:
            print("  ⚠️  Boundary case, expanding search...")
        distances = np.abs(abiLat - lat) + np.abs(abiLong - lon_normalized)
        coords = np.array(np.unravel_index(distances.argmin(), distances.shape), dtype=int)
    else:
        coords[0] += lati - AREA_SIZE
        coords[1] += loni - AREA_SIZE

    if validate_chip_region and (
        coords[0] < BOUND_SIZE or coords[1] < BOUND_SIZE or
        coords[1] > LENGTH-BOUND_SIZE or coords[0] > LENGTH-BOUND_SIZE
    ):
        raise ValueError(
            f"Chip at coords [{coords[0]}, {coords[1]}] would extend beyond valid region"
        )

    matched_lat = abiLat[coords[0], coords[1]]
    matched_lon = abiLong[coords[0], coords[1]]
    distance_km = float(np.sqrt((matched_lat - lat)**2 + (matched_lon - lon_normalized)**2) * 111)

    return {
        'coords': coords,
        'matched_lat': float(matched_lat),
        'matched_lon': float(matched_lon),
        'distance_km': distance_km,
        'lon_normalized': float(lon_normalized),
    }


def find_abi_files(year: str, doy: str, hour: str, root_path: str) -> dict:
    """
    Find all ABI channel files for a given time.
    
    Args:
        year: Year (YYYY format)
        doy: Day of year (DDD format, 001-366)
        hour: Hour (HH format, 00-23)
        root_path: Root directory for ABI data
    
    Returns:
        Dictionary with minute keys and list of channel files
    
    Raises:
        FileNotFoundError: If data path doesn't exist
    """
    abi_files = {
        "00": [], "10": [], "20": [], "30": [], "40": [], "50": [],
        "15": [], "45": [],  # Also support 15-min intervals
        "year": year, "doy": doy, "hour": hour
    }
    
    path = os.path.join(root_path, year, doy, hour)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"ABI data path not found: {path}")
    
    for filename in os.listdir(path):
        if filename.startswith("OR_ABI-L1b-RadF") and filename.endswith(".nc"):
            try:
                minute = filename[36:38]
                if minute in abi_files:
                    abi_files[minute].append(filename)
            except IndexError:
                continue
    
    return abi_files


def load_abi_channels(
    file_list: list, 
    year: str, 
    doy: str, 
    hour: str, 
    root_path: str
) -> np.ndarray:
    """
    Load and stack all 16 ABI channels to common resolution.
    
    Args:
        file_list: List of 16 channel filenames
        year, doy, hour: Time components for path
        root_path: Root directory
    
    Returns:
        Stacked array of shape (10848, 10848, 16)
    
    Raises:
        ValueError: If not exactly 16 channels found
    """
    if len(file_list) != 16:
        raise ValueError(f"Expected 16 channels, found {len(file_list)}")
    
    channels = []
    path = os.path.join(root_path, year, doy, hour)
    
    for filename in file_list:
        # Extract channel number
        channel_num = int(filename[19:21])
        
        # Load radiance data
        filepath = os.path.join(path, filename)
        try:
            with nc.Dataset(filepath, 'r') as f:
                rad = f['Rad'][:]
        except Exception as e:
            raise RuntimeError(f"Error loading {filename}: {e}")
        
        channels.append((rad, channel_num))
    
    # Sort by channel number
    channels.sort(key=lambda x: x[1])
    channels = [c[0] for c in channels]
    
    # Resample to common 10848×10848 grid
    resampled = []
    for channel_data in channels:
        shape_factor = channel_data.shape[0] // 5424
        
        if shape_factor == 1:
            # Upsample 2x
            channel_data = np.repeat(channel_data, 2, axis=0)
            channel_data = np.repeat(channel_data, 2, axis=1)
        elif shape_factor == 4:
            # Downsample 2x
            channel_data = channel_data[::2, ::2]
        
        resampled.append(channel_data)
    
    # Stack along channel dimension
    abi_stack = np.stack(resampled, axis=2)
    
    return abi_stack


def load_abi_frame_at_time(
    utc_time: datetime,
    root_path: str = ABIDATA_ROOT,
    verbose: bool = False,
) -> Dict[str, object]:
    """
    Load the ABI full-disk multi-channel frame for a UTC time.

    Returns:
        Dict with keys:
        {'abi_full', 'year', 'doy', 'hour', 'minute', 'abi_files'}
    """
    year, doy, hour, minute = parse_utc_to_abi_time(utc_time)

    try:
        abi_files = find_abi_files(year, doy, hour, root_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"ABI data directory not found: {root_path}/{year}/{doy}/{hour}/"
        ) from e

    if not abi_files[minute]:
        available = [k for k, v in abi_files.items() if v and k.isdigit()]
        raise FileNotFoundError(
            f"No ABI files for {year}/{doy} {hour}:{minute}. "
            f"Available: {available}"
        )

    if verbose:
        print(f"  Found {len(abi_files[minute])} channel files")
        print(f"\n Loading ABI channels...")

    try:
        abi_full = load_abi_channels(abi_files[minute], year, doy, hour, root_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load ABI data: {e}") from e

    if verbose:
        print(f"  Loaded full disk: {abi_full.shape}")

    return {
        'abi_full': abi_full,
        'year': year,
        'doy': doy,
        'hour': hour,
        'minute': minute,
        'abi_files': abi_files,
    }


def build_chip_data_from_coords(
    abi_full: np.ndarray,
    center_coords: Union[np.ndarray, Tuple[int, int]],
    chip_half_size: int = DEFAULT_CHIP_HALF_SIZE,
    validate_chip_region: bool = True,
) -> Dict[str, object]:
    """
    Extract a chip and geolocation metadata from a center ABI pixel coordinate.

    Args:
        abi_full: Full-disk ABI stack (H, W, C)
        center_coords: ABI pixel center as (row, col)
        chip_half_size: Half chip size in pixels
        validate_chip_region: Enforce BOUND_SIZE and image bound checks

    Returns:
        chip_data dict with keys:
        {'chip', 'lat_chip', 'lon_chip', 'coords', 'lat_transect', 'lon_transect',
         'center_lat', 'center_lon'}
    """
    if not _geolocation_loaded:
        load_geolocation_grid()

    coords = np.array([int(center_coords[0]), int(center_coords[1])], dtype=int)
    row_start = coords[0] - chip_half_size
    row_end = coords[0] + chip_half_size
    col_start = coords[1] - chip_half_size
    col_end = coords[1] + chip_half_size

    if validate_chip_region:
        if coords[0] < BOUND_SIZE or coords[1] < BOUND_SIZE or \
           coords[1] > LENGTH-BOUND_SIZE or coords[0] > LENGTH-BOUND_SIZE:
            raise ValueError(
                f"Center {tuple(coords)} outside valid chip region"
            )
        if row_start < 0 or col_start < 0 or row_end > LENGTH or col_end > LENGTH:
            raise ValueError(
                f"Chip bounds out of full-disk range for center {tuple(coords)}"
            )

    chip = abi_full[row_start:row_end, col_start:col_end, :]
    lat_chip = abiLat[row_start:row_end, col_start:col_end]
    lon_chip = abiLong[row_start:row_end, col_start:col_end]
    center_lat = float(abiLat[coords[0], coords[1]])
    center_lon = float(abiLong[coords[0], coords[1]])
    lat_transect = np.linspace(center_lat - 0.424, center_lat + 0.424, 91)
    lon_transect = np.linspace(center_lon - 0.118, center_lon + 0.118, 91)

    return {
        'chip': chip,
        'lat_chip': lat_chip,
        'lon_chip': lon_chip,
        'coords': coords,
        'center_lat': center_lat,
        'center_lon': center_lon,
        'lat_transect': lat_transect,
        'lon_transect': lon_transect,
    }


def generate_abi_chip(
    lat: float,
    lon: float,
    utc_time: datetime,
    chip_half_size: int = DEFAULT_CHIP_HALF_SIZE,
    root_path: str = ABIDATA_ROOT,
    return_chip_data: bool = True,
    verbose: bool = True
) -> Union[np.ndarray, Dict[str, object]]:
    """
    Generate an ABI chip centered at given coordinates and time.
    
    Args:
        lat: Latitude in degrees (-90 to 90)
        lon: Longitude in degrees (0 to 360 or -180 to 180)
        utc_time: UTC datetime of desired ABI image
        chip_half_size: Half-size of chip (default 64 → 128×128)
        root_path: Root directory for ABI L1b data
        return_chip_data: If True, return dict with chip/lat/lon arrays and coords
        verbose: Print progress messages
    
    Returns:
        - If return_chip_data=False: chip array (H, W, 16)
        - If return_chip_data=True: chip_data dict with keys
          {'chip', 'lat_chip', 'lon_chip', 'coords', 'lat_transect', 'lon_transect'}
    
    Raises:
        ValueError: If coordinates out of bounds
        FileNotFoundError: If ABI data not found
        RuntimeError: If error loading data
    
    Example:
        >>> from datetime import datetime
        >>> chip = generate_abi_chip(35.5, -95.0, datetime(2019, 9, 18, 20, 0))
        >>> print(chip.shape)
        (128, 128, 16)
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🎯 Generating ABI Chip")
        print(f"{'='*70}")
        print(f"  Center: ({lat:.4f}°, {lon:.4f}°)")
        print(f"  Time: {utc_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  Chip size: {2*chip_half_size}×{2*chip_half_size} pixels")
    
    # Step 1: Find nearest ABI pixel
    if verbose:
        print(f"\n📍 Finding nearest ABI pixel...")
    match = find_nearest_abi_pixel(lat=lat, lon=lon, validate_chip_region=True, verbose=verbose)
    coords = match['coords']
    matched_lat = match['matched_lat']
    matched_lon = match['matched_lon']
    distance_km = match['distance_km']
    
    if verbose:
        print(f"     Matched pixel: row={coords[0]}, col={coords[1]}")
        print(f"     Match error: ~{distance_km:.2f} km")
    
    # Step 2: Parse time and find ABI files
    if verbose:
        print(f"\n Locating ABI files...")
    
    frame = load_abi_frame_at_time(utc_time=utc_time, root_path=root_path, verbose=verbose)
    abi_full = frame['abi_full']
    
    # Step 4: Extract chip
    if verbose:
        print(f"\n Extracting chip...")
    
    chip_data = build_chip_data_from_coords(
        abi_full=abi_full,
        center_coords=coords,
        chip_half_size=chip_half_size,
        validate_chip_region=True,
    )
    chip = chip_data['chip']
    
    if verbose:
        print(f"     Extracted: {chip.shape}")
        print(f"     Radiance range: [{chip.min():.2f}, {chip.max():.2f}]")
        print(f"     Size: {chip.nbytes / 1024**2:.2f} MB")
        print(f"\n{'='*70}")
        print(f"✅ Chip generation complete!")
        print(f"{'='*70}\n")
    
    if return_chip_data:
        # Preserve matched-point values from the lat/lon search while reusing
        # the shared chip extraction helper for arrays and center-based metadata.
        chip_data['center_lat'] = matched_lat
        chip_data['center_lon'] = matched_lon
        chip_data['lat_transect'] = np.linspace(matched_lat - 0.424, matched_lat + 0.424, 91)
        chip_data['lon_transect'] = np.linspace(matched_lon - 0.118, matched_lon + 0.118, 91)
        return chip_data
    return chip


def batch_generate_chips(
    utc_time: datetime,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    coords: Optional[Tuple[int, int]] = None,
    chip_half_size: int = DEFAULT_CHIP_HALF_SIZE,
    max_offset: Optional[int] = None,
    stride_size: int = 4,
    root_path: str = ABIDATA_ROOT,
    verbose: bool = True,
) -> list:
    """
    Generate a batch of chips by sliding along x (column) direction.

    A single reference center is provided either as:
    - (lat, lon), which will be matched to the nearest ABI pixel, or
    - coords=(row, col), which is used directly.

    Chips are then extracted at offsets along the same row:
    dx = -max_offset, -max_offset+stride_size, ..., +max_offset

    Args:
        utc_time: UTC datetime for ABI image
        lat: Latitude of reference center (used with lon)
        lon: Longitude of reference center (used with lat)
        coords: ABI pixel coordinates (row, col) of reference center
        chip_half_size: Half-size of chips
        max_offset: Maximum x-offset in pixels from reference center
            (default: chip_half_size)
        stride_size: Pixel step along x direction (default 4)
        root_path: Root directory for ABI data
        verbose: Print progress messages

    Returns:
        List of per-chip dicts. Each item includes:
        {'offset_x', 'chip', 'lat_chip', 'lon_chip', 'coords',
         'lat_transect', 'lon_transect', 'center_lat', 'center_lon',
         'success', 'error'}

    Example:
        >>> results = batch_generate_chips(
        ...     utc_time=datetime(2019, 9, 18, 20, 0),
        ...     lat=35.5, lon=-95.0,
        ...     chip_half_size=64, max_offset=64, stride_size=4
        ... )
        >>> print(len(results))
    """
    if max_offset is None:
        max_offset = chip_half_size
    if max_offset < 0:
        raise ValueError(f"max_offset must be >= 0, got {max_offset}")
    if stride_size <= 0:
        raise ValueError(f"stride_size must be > 0, got {stride_size}")

    use_coords = coords is not None
    use_latlon = lat is not None or lon is not None
    if use_coords and use_latlon:
        raise ValueError("Provide either coords or (lat, lon), not both")
    if not use_coords and not (lat is not None and lon is not None):
        raise ValueError("Provide coords=(row, col) or both lat and lon")

    if not _geolocation_loaded:
        load_geolocation_grid()

    # Determine reference center in ABI pixel coordinates.
    if use_coords:
        base_coords = np.array([int(coords[0]), int(coords[1])], dtype=int)
    else:
        match = find_nearest_abi_pixel(lat=lat, lon=lon, validate_chip_region=False, verbose=verbose)
        base_coords = match['coords']

    if verbose:
        print(f"\n{'='*70}")
        print("🔄 Batch X-Direction Chip Generation")
        print(f"{'='*70}")
        print(f"  Reference center (row, col): {tuple(base_coords)}")
        print(f"  Time: {utc_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  Chip size: {2*chip_half_size}×{2*chip_half_size}")
        print(f"  X offsets: {-max_offset} to +{max_offset} (step {stride_size})")

    # Load ABI data once for all shifted chips.
    frame = load_abi_frame_at_time(utc_time=utc_time, root_path=root_path, verbose=False)
    abi_full = frame['abi_full']

    offsets = list(range(-max_offset, max_offset + 1, stride_size))
    results = []

    for i, dx in enumerate(offsets):
        center_coords = np.array([base_coords[0], base_coords[1] + dx], dtype=int)
        if verbose:
            print(f"[{i+1}/{len(offsets)}] offset_x={dx:+d}, center={tuple(center_coords)}")

        try:
            chip_data = build_chip_data_from_coords(
                abi_full=abi_full,
                center_coords=center_coords,
                chip_half_size=chip_half_size,
                validate_chip_region=True,
            )

            results.append({
                'index': i,
                'offset_x': dx,
                **chip_data,
                'success': True,
                'error': None,
            })
            if verbose:
                print("  ✅ Success")
        except Exception as e:
            results.append({
                'index': i,
                'offset_x': dx,
                'chip': None,
                'lat_chip': None,
                'lon_chip': None,
                'coords': center_coords,
                'center_lat': None,
                'center_lon': None,
                'lat_transect': None,
                'lon_transect': None,
                'success': False,
                'error': str(e),
            })
            if verbose:
                print(f"  ❌ Failed: {e}")

    if verbose:
        success_count = sum(1 for r in results if r['success'])
        print(f"\n{'='*70}")
        print(f"Completed: {success_count}/{len(results)} successful")
        print(f"{'='*70}\n")

    return results


# ============================================================================
# Main execution example
# ============================================================================

def main(
    year: str = "2019",
    doy: str = "261",
    hour_min: str = "0800",
    lat: float = 35.5,
    lon: float = -95.0,
    output_root: str = OUTPUT_ROOT,
) -> None:
    """
    Example entrypoint for generating a batch of ABI chips (x-direction scan).

    Args:
        year: Year in YYYY format (e.g., "2019")
        doy: Day-of-year in DDD format (e.g., "261")
        hour_min: Time in HHMM format (e.g., "0800")
        lat: Latitude in degrees
        lon: Longitude in degrees
        output_root: Directory where output .npz is saved
    """
    year_str = str(year)
    doy_str = str(doy)
    hour_min_str = str(hour_min)

    if not (year_str.isdigit() and len(year_str) == 4):
        raise ValueError(f"year must be YYYY, got {year!r}")
    if not (doy_str.isdigit() and len(doy_str) == 3):
        raise ValueError(f"doy must be DDD, got {doy!r}")
    if not (hour_min_str.isdigit() and len(hour_min_str) == 4):
        raise ValueError(f"hour_min must be HHMM, got {hour_min!r}")

    print("\n" + "="*70)
    print("GOES-16 ABI Chip Generator")
    print("="*70)

    utc_time = datetime.strptime(f"{year_str}-{doy_str} {hour_min_str}", "%Y-%j %H%M")

    try:
        results = batch_generate_chips(
            utc_time=utc_time,
            lat=lat,
            lon=lon,
            chip_half_size=64,
            max_offset=64,
            stride_size=4,
            verbose=True,
        )

        success_results = [r for r in results if r["success"]]

        print(f"\n📊 Result Summary:")
        print(f"   Input center: ({lat:.4f}, {lon:.4f})")
        print(f"   UTC time: {utc_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Chips requested: {len(results)}")
        print(f"   Chips saved: {len(success_results)}")
        if success_results:
            sample = success_results[len(success_results) // 2]
            chip = sample["chip"]
            coords = sample["coords"]
            lat_chip = sample["lat_chip"]
            lat_transect = sample["lat_transect"]
            lon_transect = sample["lon_transect"]
            print(f"   Sample chip shape: {chip.shape}")
            print(f"   Sample dtype: {chip.dtype}")
            print(f"   Sample memory: {chip.nbytes / 1024**2:.2f} MB")
            print(f"   Sample lat/lon grid shape: {lat_chip.shape}")
            print(f"   Sample center pixel (row, col): {tuple(coords)}")
            print(f"   Sample transect points: {lat_transect.shape[0]}")
            print(f"   Sample transect lat range: [{lat_transect[0]:.4f}, {lat_transect[-1]:.4f}]")
            print(f"   Sample transect lon range: [{lon_transect[0]:.4f}, {lon_transect[-1]:.4f}]")

        os.makedirs(output_root, exist_ok=True)
        utc_tag = utc_time.strftime("%Y%j_%H%M")
        for r in success_results:
            row, col = int(r["coords"][0]), int(r["coords"][1])
            output_file = f"abi_chip_{utc_tag}_r{row:05d}_c{col:05d}.npz"
            output_path = os.path.join(output_root, output_file)
            np.savez(
                output_path,
                **{k: v for k, v in r.items() if k not in ["index", "success", "error"]}
            )
            print(f"💾 Saved: {output_path}")

    except Exception as e:
        print(f"\n❌ Error: {e}")

    print("\n" + "="*70)
    print("Done!")
    print("="*70 + "\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for the example entrypoint."""
    parser = argparse.ArgumentParser(
        description="Generate a GOES-16 ABI chip and save outputs to example_chip.npz"
    )
    parser.add_argument("--year", default="2019", help="Year in YYYY format (default: 2019)")
    parser.add_argument("--doy", default="261", help="Day-of-year in DDD format (default: 261)")
    parser.add_argument(
        "--hour-min",
        dest="hour_min",
        default="0800",
        help="UTC time in HHMM format (default: 0800)",
    )
    parser.add_argument("--lat", type=float, default=35.5, help="Latitude (default: 35.5)")
    parser.add_argument("--lon", type=float, default=-95.0, help="Longitude (default: -95.0)")
    parser.add_argument(
        "--output-root",
        default=OUTPUT_ROOT,
        help=f"Output directory for .npz files (default: {OUTPUT_ROOT})",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    main(
        year=args.year,
        doy=args.doy,
        hour_min=args.hour_min,
        lat=args.lat,
        lon=args.lon,
        output_root=args.output_root,
    )
