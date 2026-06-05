import numpy as np
import pyhdf
from pyhdf.SD import SD, SDC
from pyhdf.HDF import *
from pyhdf.VS import *
from datetime import datetime, timezone
from scipy import interpolate
import glob
import os
import matplotlib.pyplot as plt
import netCDF4 as nc
from geopy.distance import geodesic
import sys

'''
CONFIGURATION PARAMETERS - PAIRED GOES-16 and GOES-17
'''

# Geolocation data for both satellites
LATLONDATA_G16 = "/explore/nobackup/projects/pix4dcloud/jgong/ABI_EAST_GEO_TOPO_LOMSK.nc"
LATLONDATA_G17 = "/explore/nobackup/projects/pix4dcloud/jgong/ABI_WEST_GEO_TOPO_LOMSK.nc"

# CloudSat data
CLOUDSATPATH = '/explore/nobackup/projects/pix4dcloud/szhang16/cloudsat/'
ROOT_DIR = '/explore/nobackup/projects/pix4dcloud/szhang16/cloudsat/2B-CLDCLASS-LIDAR'

# ABI data paths for both satellites
ABIDATA_G16 = "/css/geostationary/NonOptimized/L1/GOES-16-ABI-L1B-FULLD/"
ABIDATA_G17 = "/css/geostationary/NonOptimized/L1/GOES-17-ABI-L1B-FULLD/"

# Output directory
SAVEDIR = '/explore/nobackup/projects/pix4dcloud/jli30/abiChips/GOES-16-17-pair-p128/'

# Create output directory if it doesn't exist
os.makedirs(SAVEDIR, exist_ok=True)

# ===============================================


def read_2b_cldclass_lidar(cs_file, latbin=None):
    # ===============================================
    f_2b_cldclass_lidar = SD(cs_file, SDC.READ)
    sds_obj = f_2b_cldclass_lidar.select('CloudLayerBase')
    cs_clb = sds_obj.get()
    sds_obj = f_2b_cldclass_lidar.select('CloudLayerTop')
    cs_clt = sds_obj.get()
    sds_obj = f_2b_cldclass_lidar.select('CloudLayerType')
    cs_cltype = sds_obj.get()

    sdc_2bcldclass_lidar = HDF(cs_file, SDC.READ)
    vs_2bcldclass_lidar = sdc_2bcldclass_lidar.vstart()
    cs_QC = np.squeeze(vs_2bcldclass_lidar.attach('Data_quality')[:])
    Latitude = np.squeeze(vs_2bcldclass_lidar.attach('Latitude')[:])
    Longitude = np.squeeze(vs_2bcldclass_lidar.attach('Longitude')[:])

    ilat = np.squeeze(np.argwhere(
        (Latitude >= latbin[0]) & (Latitude <= latbin[1])))
    cs_clb = cs_clb[ilat, :]
    cs_clt = cs_clt[ilat, :]
    cs_QC = cs_QC[ilat]
    cs_cltype = cs_cltype[ilat, :]
    Latitude = Latitude[ilat]
    Longitude = Longitude[ilat]

    return (cs_clb, cs_clt, cs_cltype, cs_QC, Latitude, Longitude)

# ===============================================


def read_cs_ecmwf(aux_file, latbin=None):
    # ===============================================
    f_ecmwf = SD(aux_file, SDC.READ)
    sds_obj = f_ecmwf.select('Pressure')
    Pressure = sds_obj.get()
    sds_obj = f_ecmwf.select('Temperature')
    Temperature = sds_obj.get()
    sds_obj = f_ecmwf.select('Specific_humidity')
    Specific_humidity = sds_obj.get()

    sdc_ecmwf = HDF(aux_file, SDC.READ)
    vs_ecmwf = sdc_ecmwf.vstart()
    EC_height = np.squeeze(vs_ecmwf.attach('EC_height')[:])
    Profile_time = np.squeeze(vs_ecmwf.attach('Profile_time')[:])
    UTC_start = np.squeeze(vs_ecmwf.attach('UTC_start')[:])
    Latitude = np.squeeze(vs_ecmwf.attach('Latitude')[:])
    Longitude = np.squeeze(vs_ecmwf.attach('Longitude')[:])
    DEM_elevation = np.squeeze(vs_ecmwf.attach('DEM_elevation')[:])
    Skin_temperature = np.squeeze(vs_ecmwf.attach('Skin_temperature')[:])
    Surface_pressure = np.squeeze(vs_ecmwf.attach('Surface_pressure')[:])
    Temperature_2m = np.squeeze(vs_ecmwf.attach('Temperature_2m')[:])
    U10_velocity = np.squeeze(vs_ecmwf.attach('U10_velocity')[:])
    V10_velocity = np.squeeze(vs_ecmwf.attach('V10_velocity')[:])

    UTC_Time = UTC_start + Profile_time
    UTC_Time = UTC_Time/60./60.
    ilat = np.squeeze(np.argwhere(
        (Latitude >= latbin[0]) & (Latitude <= latbin[1])))
    Pressure = Pressure[ilat, :]
    Temperature = Temperature[ilat, :]
    Specific_humidity = Specific_humidity[ilat, :]
    DEM_elevation = DEM_elevation[ilat]
    Skin_temperature = Skin_temperature[ilat]
    Temperature_2m = Temperature_2m[ilat]
    U10_velocity = U10_velocity[ilat]
    V10_velocity = V10_velocity[ilat]
    UTC_Time = UTC_Time[ilat]

    return (Pressure, DEM_elevation, Temperature, Specific_humidity, EC_height, Temperature_2m, U10_velocity, V10_velocity, UTC_Time)


def find_files(directory, prefix):
    matching_files = []
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            matching_files.append(os.path.join(directory, filename))
    return matching_files


def gather_files(YYYY, DDD, HH, ROOT):
    ABI_ = {
        "ROOT_PATH": None,

        "YYYY": None,
        "DDD": None,
        "HH": None,

        "00": [],
        "10": [],
        "15": [],
        "20": [],
        "30": [],
        "40": [],
        "45": [],
        "50": [],

        "L200": None,
        "L210": None,
        "L220": None,
        "L230": None,
        "L240": None,
        "L250": None,

        "everyten": False,
    }

    _ABI_PATH_ = ROOT + YYYY + "/" + DDD + "/" + HH

    if not os.path.exists(_ABI_PATH_):
        return None

    for filename in os.listdir(_ABI_PATH_):
        if ABI_["ROOT_PATH"] == None:
            ABI_["ROOT_PATH"] = _ABI_PATH_
            ABI_["YYYY"] = filename[27:31]
            ABI_["DDD"] = filename[31:34]
            ABI_["HH"] = filename[34:36]
        MM = filename[36:38]
        if MM == "10":
            ABI_["everyten"] = True
        ABI_[f"{MM}"].append(filename)

    return ABI_


def get_L1B_L2(abipaths, l2path, YYYY, DDD, HH, ROOT):

    if len(abipaths) != 16:
        raise ImportError("This hour is bad")

    CHANNELS = []

    for file in abipaths:
        L1B = np.asarray(nc.Dataset(ROOT + YYYY + "/" + DDD +
                         "/" + HH + "/" + file, 'r')["Rad"])
        CHANNEL = int(file[19:21])
        CHANNELS.append((L1B, CHANNEL))

    CHANNELS.sort(key=lambda x: x[1])
    CHANNELS = [C[0] for C in CHANNELS]

    T = []
    for C in CHANNELS:
        S = C.shape[0] // 5424
        if S == 1:
            C = np.repeat(C, 2, axis=0)
            C = np.repeat(C, 2, axis=1)
        if S == 4:
            C = C[::2, ::2]
        T.append(C)

    CHANNELS = T

    ABI = np.stack(CHANNELS, axis=2)

    return ABI


class SatelliteGrid:
    """Class to handle lat/lon grid for a satellite"""
    def __init__(self, latlon_file, name):
        self.name = name
        f = nc.Dataset(latlon_file)
        self.abiLong = np.array(f['Longitude'])
        self.abiLat = np.array(f['Latitude'])
        
        BOUND_SIZE = 1600
        LENGTH = 10848
        
        self.abiLongB = self.abiLong[BOUND_SIZE:LENGTH-BOUND_SIZE, BOUND_SIZE:LENGTH-BOUND_SIZE]
        self.abiLatB = self.abiLat[BOUND_SIZE:LENGTH-BOUND_SIZE, BOUND_SIZE:LENGTH-BOUND_SIZE]
        self.abiLongB[self.abiLongB == -999] = 10
        self.abiLatB[self.abiLatB == -999] = 10
        self.abiLongB[self.abiLongB < 0] += 360
        
        self.longMin = self.abiLongB.min()
        self.longMax = self.abiLongB.max()
        self.latMin = self.abiLatB.min()
        self.latMax = self.abiLatB.max()
        
        self.latSlice = self.abiLat[:, 5424]
        self.latSlice = self.latSlice[18:-18]
        self.longSlice = self.abiLong[5424, :]
        self.longSlice = self.longSlice[18:-18]
        self.latSlice = self.latSlice[::-1]
        
        self.BOUND_SIZE = BOUND_SIZE
        self.LENGTH = LENGTH
        
    def is_in_bounds(self, lat, lon):
        """Check if lat/lon is within valid bounds"""
        return (lat >= self.latMin and lat <= self.latMax and 
                lon >= self.longMin and lon <= self.longMax)
    
    def find_pixel_coords(self, lat, lon):
        """Find pixel coordinates for given lat/lon"""
        if not self.is_in_bounds(lat, lon):
            return None
            
        AREA_SIZE = 1000
        
        lati = len(self.latSlice) - np.searchsorted(self.latSlice, lat) + 17
        loni = np.searchsorted(self.longSlice, lon) + 18
        
        distances = np.abs(self.abiLat[lati-AREA_SIZE:lati+AREA_SIZE, loni-AREA_SIZE:loni+AREA_SIZE] - lat) + \
            np.abs(self.abiLong[lati-AREA_SIZE:lati+AREA_SIZE, loni-AREA_SIZE:loni+AREA_SIZE] - lon)
        coords = np.array(np.unravel_index(distances.argmin(), distances.shape))
        
        if coords[0] == 0 or coords[1] == 0 or coords[1] == 2*AREA_SIZE-1 or coords[0] == 2*AREA_SIZE - 1:
            distances = np.abs(self.abiLat - lat) + np.abs(self.abiLong - lon)
            coords = np.unravel_index(distances.argmin(), distances.shape)
        else:
            coords[0] += lati - AREA_SIZE
            coords[1] += loni - AREA_SIZE
        
        if coords[0] < self.BOUND_SIZE or coords[1] < self.BOUND_SIZE or \
           coords[1] > self.LENGTH-self.BOUND_SIZE or coords[0] > self.LENGTH-self.BOUND_SIZE:
            return None
            
        return coords


# Initialize grids for both satellites
print("Loading satellite grids...")
GRID_G16 = SatelliteGrid(LATLONDATA_G16, "GOES-16")
GRID_G17 = SatelliteGrid(LATLONDATA_G17, "GOES-17")

# Find overlap region
OVERLAP_LAT_MIN = max(GRID_G16.latMin, GRID_G17.latMin)
OVERLAP_LAT_MAX = min(GRID_G16.latMax, GRID_G17.latMax)
OVERLAP_LON_MIN = max(GRID_G16.longMin, GRID_G17.longMin)
OVERLAP_LON_MAX = min(GRID_G16.longMax, GRID_G17.longMax)

print(f"GOES-16 Coverage: Lat [{GRID_G16.latMin:.2f}, {GRID_G16.latMax:.2f}], Lon [{GRID_G16.longMin:.2f}, {GRID_G16.longMax:.2f}]")
print(f"GOES-17 Coverage: Lat [{GRID_G17.latMin:.2f}, {GRID_G17.latMax:.2f}], Lon [{GRID_G17.longMin:.2f}, {GRID_G17.longMax:.2f}]")
print(f"Overlap Region: Lat [{OVERLAP_LAT_MIN:.2f}, {OVERLAP_LAT_MAX:.2f}], Lon [{OVERLAP_LON_MIN:.2f}, {OVERLAP_LON_MAX:.2f}]")

CHIP_HALF_SIZE = 64  # 128x128 chips

# Separate caches for each satellite
GATHERED_ABI_FILES_G16 = {}
COLLECTED_ABI_DATA_G16 = {}
GATHERED_ABI_FILES_G17 = {}
COLLECTED_ABI_DATA_G17 = {}


def interpArray(Temperature, EC_height):
    z_grid = np.arange(40)*0.5
    N = Temperature.shape[0]
    Temperature_grid = np.zeros((N, len(z_grid)))
    for i in range(N):
        Temperature_tmp = np.squeeze(Temperature[i, :])
        ivalid = np.squeeze(np.where(Temperature_tmp > 0))
        Temperature_grid[i, :] = np.interp(z_grid, np.flip(
            EC_height[ivalid]/1000.), np.flip(Temperature_tmp[ivalid]))
    return Temperature_grid


def processTime_satellite(t, yy, ddn, lat, lon, ABI_ROOT, GRID, GATHERED_FILES, COLLECTED_DATA, sat_name):
    """Process time for a specific satellite"""
    
    if np.floor(t) < 19:
        raise ValueError("outside 19-23 bound")

    if np.floor(t) == 24:
        raise ValueError("24 hour")

    # Check if location is in this satellite's coverage
    coords = GRID.find_pixel_coords(lat, lon)
    if coords is None:
        raise ValueError(f"Location not in {sat_name} coverage")

    hour = np.floor(t).astype(int)
    if hour < 10:
        hour = "0" + str(hour)

    DATA = GATHERED_FILES.get(f'{yy}-{ddn}-{hour}')
    if DATA is None:
        DATA = gather_files(str(yy), str(ddn), str(hour), ABI_ROOT)
        if DATA is None:
            raise ImportError(f"No {sat_name} data for this hour")
        GATHERED_FILES[f'{yy}-{ddn}-{hour}'] = DATA

    if DATA["everyten"]:
        minutes = np.round((t - np.floor(t)) * 6).astype(int) * 10
    else:
        minutes = np.round((t - np.floor(t)) * 4).astype(int) * 15

    if minutes == 60:
        if hour != "23":
            hour = str(int(hour) + 1)
            if int(hour) < 10:
                hour = "0" + hour
            minutes = 0
        else:
            if DATA["everyten"]:
                minutes = 50
            else:
                minutes = 45

    minutes = str(minutes)
    if minutes == "0":
        minutes = "00"

    ABI = COLLECTED_DATA.get(f'{yy}-{ddn}-{hour}-{minutes}')
    if ABI is None:
        ABI = get_L1B_L2(DATA[minutes], DATA["L200"],
                         DATA["YYYY"], DATA["DDD"], DATA["HH"], ABI_ROOT)
        COLLECTED_DATA[f'{yy}-{ddn}-{hour}-{minutes}'] = ABI

    chip = ABI[coords[0]-CHIP_HALF_SIZE:coords[0]+CHIP_HALF_SIZE,
               coords[1]-CHIP_HALF_SIZE:coords[1]+CHIP_HALF_SIZE, :]

    return chip, coords


def processTime_paired(t, yy, ddn, lat, lon):
    """Process time for both GOES-16 and GOES-17"""
    
    # Check if location is in overlap region
    if not (OVERLAP_LAT_MIN <= lat <= OVERLAP_LAT_MAX and 
            OVERLAP_LON_MIN <= lon <= OVERLAP_LON_MAX):
        raise ValueError("Location not in overlap region")
    
    # Try to get data from both satellites
    chip_g16, coords_g16 = processTime_satellite(
        t, yy, ddn, lat, lon, ABIDATA_G16, GRID_G16, 
        GATHERED_ABI_FILES_G16, COLLECTED_ABI_DATA_G16, "GOES-16"
    )
    
    chip_g17, coords_g17 = processTime_satellite(
        t, yy, ddn, lat, lon, ABIDATA_G17, GRID_G17,
        GATHERED_ABI_FILES_G17, COLLECTED_ABI_DATA_G17, "GOES-17"
    )
    
    return chip_g16, coords_g16, chip_g17, coords_g17


def processFile(yy, ddn, orbit, latb):
    cs_file = glob.glob(CLOUDSATPATH+'2B-CLDCLASS-LIDAR/'+yy+'/' +
                        ddn+'/'+yy+ddn+'*'+orbit+'*2B-CLDCLASS-LIDAR*'+'P1_R05*.hdf')
    aux_file = glob.glob(CLOUDSATPATH+'ECMWF-AUX/'+yy+'/' +
                         ddn+'/'+yy+ddn+'*'+orbit+'*ECMWF-AUX*'+'P1_R05*.hdf')

    if len(cs_file) == 0 or len(aux_file) == 0:
        return

    [Pressure, DEM_elevation, Temperature, Specific_humidity, EC_height, Temperature_2m,
        U10_velocity, V10_velocity, UTC_Time] = read_cs_ecmwf(aux_file[0], latbin=latb)

    # ========== read 2b-cldclass-lidar ================================
    [cs_clb, cs_clt, cs_cltype, cs_QC, Latitude,
        Longitude] = read_2b_cldclass_lidar(cs_file[0], latbin=latb)
    Longitude[Longitude < 0] += 360

    Pressure = interpArray(Pressure, EC_height)
    Temperature = interpArray(Temperature, EC_height)
    Specific_humidity = interpArray(Specific_humidity, EC_height)

    N = len(cs_clb)

    cs_clb_2 = np.floor(2 * cs_clb).astype(int)
    cs_clt_2 = np.floor(2 * cs_clt).astype(int)

    cloud_class_mask_40_level = np.zeros((N, 40), dtype=np.int8)

    for i in range(N):
        for j in range(10):
            if cs_clb[i, j] < 0:
                break
            start_idx = cs_clb_2[i, j]
            end_idx = cs_clt_2[i, j] + 1
            cloud_type = cs_cltype[i, j]
            cloud_class_mask_40_level[i, start_idx:end_idx] = cloud_type

    i = 0
    HALF_FOOTPRINTS = 45  # 91 total footprints
    TOTAL_FOOTPRINTS = 2 * HALF_FOOTPRINTS + 1
    
    paired_count = 0
    skipped_count = 0

    while i < N:
        try:
            chip_g16, coords_g16, chip_g17, coords_g17 = processTime_paired(
                UTC_Time[i], yy, ddn, Latitude[i], Longitude[i]
            )
        except ValueError as e:
            i += 20
            skipped_count += 1
            continue
        except ImportError as e:
            i += 90
            skipped_count += 1
            continue
        except Exception as e:
            i += 20
            skipped_count += 1
            continue

        if i + HALF_FOOTPRINTS + 1 >= N:
            break

        # Reverse the CloudSAT Data
        dRange = np.arange(i + HALF_FOOTPRINTS + 1, i - HALF_FOOTPRINTS, -1)
        aux_data = {
            "Pressure": Pressure[dRange],
            "DEM_elevation": DEM_elevation[dRange],
            "Temperature": Temperature[dRange],
            "Specific_humidity": Specific_humidity[dRange],
            "Temperature_2m": Temperature_2m[dRange],
            "U10_velocity": U10_velocity[dRange],
            "V10_velocity": V10_velocity[dRange],
            "UTC_Time": UTC_Time[dRange],
            "Latitude": Latitude[dRange],
            "Longitude": Longitude[dRange],
            "Cloud_mask": cloud_class_mask_40_level[dRange],
            "Cloud_mask_binary": (cloud_class_mask_40_level[dRange] != 0).astype(int),
            "Cloud_class": cs_cltype[dRange],
        }

        fileName = f'{yy}-{ddn}-{orbit}_G16-{coords_g16[0]}-{coords_g16[1]}_G17-{coords_g17[0]}-{coords_g17[1]}_i{i}'

        np.savez(SAVEDIR + fileName, 
                 chip_g16=chip_g16, 
                 chip_g17=chip_g17,
                 coords_g16=coords_g16,
                 coords_g17=coords_g17,
                 data=aux_data)
        
        paired_count += 1
        print(f"{fileName} | Time: {UTC_Time[i]:.2f} | Paired: {paired_count} | Skipped: {skipped_count}")
        i += HALF_FOOTPRINTS
    
    print(f"Orbit {orbit} complete: {paired_count} paired chips saved, {skipped_count} skipped")

# =================================
#    MAIN
# =================================

year = 0
dayskip = 0
try:
    year = sys.argv[1]
    dayskip = sys.argv[2]
except:
    print("Pass in year as the argument")
    print("Usage: python cropChipsPaired.py <year> <dayskip>")
    sys.exit(1)

for day in os.listdir(ROOT_DIR + '/' + year):
    if int(day) < int(dayskip):
        continue
    for file in os.listdir(ROOT_DIR + '/' + year + '/' + day):
        if file.endswith('hdf'):
            orbit = file[14:19]
            print(f"\n{'='*60}")
            print(f"PROCESSING FILE {year} {day} {orbit}")
            print(f"{'='*60}")
            processFile(year, day, orbit, (OVERLAP_LAT_MIN, OVERLAP_LAT_MAX))
    
    # Reset ABI data caches after each day
    GATHERED_ABI_FILES_G16 = {}
    COLLECTED_ABI_DATA_G16 = {}
    GATHERED_ABI_FILES_G17 = {}
    COLLECTED_ABI_DATA_G17 = {}
    print(f"\nDay {day} complete. Caches cleared.\n")

print("\n" + "="*60)
print("PROCESSING COMPLETE")
print("="*60)