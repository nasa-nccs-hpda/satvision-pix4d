import os
import sys
import glob
import pyhdf
import logging
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

from pyhdf.VS import *
from pyhdf.HDF import *
from pyhdf.SD import SD, SDC
from scipy import interpolate
from geopy.distance import geodesic
from datetime import datetime, timezone


class ABICloudSatCropping:
    def __init__(
        self,
        latlon_data: str = "/explore/nobackup/projects/pix4dcloud/jgong/ABI_EAST_GEO_TOPO_LOMSK.nc",
        root_dir: str = "/explore/nobackup/projects/pix4dcloud/szhang16/cloudsat/2B-CLDCLASS-LIDAR",
        cloudsat_path: str = "/explore/nobackup/projects/pix4dcloud/szhang16/cloudsat",
        abidata_path: str = "/css/geostationary/BackStage/GOES-16-ABI-L1B-FULLD",
        output_dir: str = "./abi_tiles",
        bound_size: int = 1600,
        length: int = 10848,
        year: str = "2019",
        dayskip: int = 0
    ):

        self.latlon_data = latlon_data
        self.root_dir = root_dir
        self.cloudsat_path = cloudsat_path
        self.abidata_path = abidata_path
        self.output_dir = output_dir

        self.year = year
        self.dayskip = dayskip

        # setup output directory
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f'Created output dir: {self.output_dir}')

        # setting up default latlon configuration
        f = nc.Dataset(self.latlon_data)
        self.abiLong = np.array(f['Longitude'])
        self.abiLat = np.array(f['Latitude'])

        self.bound_size = bound_size
        self.length = length

        self.abiLongB = self.abiLong[
            bound_size:length-bound_size, bound_size:length-bound_size]
        self.abiLatB = self.abiLat[
            bound_size:length-bound_size, bound_size:length-bound_size]
        self.abiLongB[self.abiLongB == -999] = 10
        self.abiLatB[self.abiLatB == -999] = 10
        self.abiLongB[self.abiLongB < 0] += 360

        self.longMin = self.abiLongB.min()
        self.longMax = self.abiLongB.max()
        self.latMin = self.abiLatB.min()
        self.latMax = self.abiLatB.max()

        self.GATHERED_ABI_FILES = {}
        self.COLLECTED_ABI_DATA = {}

        self.latSlice = self.abiLat[:, 5424]
        self.latSlice = self.latSlice[18:-18]
        self.longSlice = self.abiLong[5424, :]
        self.longSlice = self.longSlice[18:-18]
        self.latSlice = self.latSlice[::-1]

    def gen_tiles(self):
        day_path = os.path.join(self.root_dir, str(self.year))
        for day in os.listdir(day_path):
            if int(day) < int(self.dayskip):
                continue
            file_path = os.path.join(day_path, str(day))
            for file in os.listdir(file_path):
                if file.endswith('hdf'):
                    orbit = file[14:19]
                    logging.info(f"PROCESSING FILE {self.year} {day} {orbit}")
                    self.processFile(
                        self.year, day, orbit,
                        (self.latMin, self.latMax)
                    )
                #break # remove after testing
            # reset abi data
            self.GATHERED_ABI_FILES = {}
            self.COLLECTED_ABI_DATA = {}

        return

    def read_2b_cldclass_lidar(self, cs_file, latbin=None):
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

    def read_cs_ecmwf(self, aux_file, latbin=None):
        # ===============================================
        f_ecmwf = SD(aux_file, SDC.READ)
        sds_obj = f_ecmwf.select('Pressure')
        Pressure = sds_obj.get()
        sds_obj = f_ecmwf.select('Temperature')
        Temperature = sds_obj.get()
        sds_obj = f_ecmwf.select('Specific_humidity')
        Specific_humidity = sds_obj.get()
        # sds_obj=f_ecmwf.select('U_velocity')
        # U_velocity =sds_obj.get()
        # sds_obj=f_ecmwf.select('V_velocity')
        # V_velocity =sds_obj.get()

        sdc_ecmwf = HDF(aux_file, SDC.READ)
        vs_ecmwf = sdc_ecmwf.vstart()
        EC_height = np.squeeze(vs_ecmwf.attach('EC_height')[:])
        Profile_time = np.squeeze(vs_ecmwf.attach('Profile_time')[:])
        UTC_start = np.squeeze(vs_ecmwf.attach('UTC_start')[:])
        # TAI_start = np.squeeze(vs_ecmwf.attach('TAI_start')[:])
        Latitude = np.squeeze(vs_ecmwf.attach('Latitude')[:])
        Longitude = np.squeeze(vs_ecmwf.attach('Longitude')[:])
        DEM_elevation = np.squeeze(vs_ecmwf.attach('DEM_elevation')[:])
        Skin_temperature = np.squeeze(vs_ecmwf.attach('Skin_temperature')[:])
        Surface_pressure = np.squeeze(vs_ecmwf.attach('Surface_pressure')[:])
        Temperature_2m = np.squeeze(vs_ecmwf.attach('Temperature_2m')[:])
        # Sea_surface_temperature = np.squeeze(vs_ecmwf.attach('Sea_surface_temperature')[:])
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

    def find_files(self, directory, prefix):
        matching_files = []
        for filename in os.listdir(directory):
            if filename.startswith(prefix):
                matching_files.append(os.path.join(directory, filename))
        return matching_files

    def gather_files(self, YYYY, DDD, HH, ROOT):
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

        # _ABI_PATH_ = ROOT + YYYY + "/" + DDD + "/" + HH
        _ABI_PATH_ = os.path.join(
            ROOT, YYYY, DDD, HH
        )

        for filename in os.listdir(_ABI_PATH_):
            if ABI_["ROOT_PATH"] is None:
                ABI_["ROOT_PATH"] = _ABI_PATH_
                ABI_["YYYY"] = filename[27:31]
                ABI_["DDD"] = filename[31:34]
                ABI_["HH"] = filename[34:36]
            MM = filename[36:38]
            if MM == "10":
                ABI_["everyten"] = True
            ABI_[f"{MM}"].append(filename)

        return ABI_

    def get_L1B_L2(self, abipaths, l2path, YYYY, DDD, HH, ROOT):

        if len(abipaths) != 16:
            raise ImportError("This hour is bad")

        CHANNELS = []

        for file in abipaths:

            filename = os.path.join(
                ROOT, YYYY, DDD, HH, file
            )

            L1B = np.asarray(
                nc.Dataset(filename, 'r')["Rad"]
            )
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

    def interpArray(self, Temperature, EC_height):

        z_grid = np.arange(40)*0.5
        N = Temperature.shape[0]
        Temperature_grid = np.zeros((N, len(z_grid)))
        for i in range(N):
            Temperature_tmp = np.squeeze(Temperature[i, :])
            ivalid = np.squeeze(np.where(Temperature_tmp > 0))

            Temperature_grid[i, :] = np.interp(z_grid, np.flip(
                EC_height[ivalid]/1000.), np.flip(Temperature_tmp[ivalid]))

        return Temperature_grid

    def processTime(self, t, yy, ddn, lat, lon, area_size: int = 1000):

        print("t ====== ", t)

        if np.floor(t) < 19:
            raise ValueError("outside 19-23 bound")

        if np.floor(t) == 24:
            raise ValueError("24 hour")

        if lat < self.latMin or lat > self.latMax or lon < self.longMin or lon > self.longMax:
            raise ValueError("Latitude and Longitude are not correctly bounded")

        lati = len(self.latSlice) - np.searchsorted(self.latSlice, lat) + 17
        loni = np.searchsorted(self.longSlice, lon) + 18

        distances = np.abs(
            self.abiLat[lati-area_size:lati+area_size, loni-area_size:loni+area_size] - lat) + \
            np.abs(self.abiLong[lati-area_size:lati+area_size,
                loni-area_size:loni+area_size] - lon)
        coords = np.array(np.unravel_index(distances.argmin(), distances.shape))
        if coords[0] == 0 or coords[1] == 0 or coords[1] == 2*area_size-1 or coords[0] == 2*area_size - 1:
            distances = np.abs(self.abiLat - lat) + np.abs(self.abiLong - lon)
            coords = np.unravel_index(distances.argmin(), distances.shape)
        else:
            coords[0] += lati - area_size
            coords[1] += loni - area_size

        if coords[0] < self.bound_size or coords[1] < self.bound_size or coords[1] > self.length-self.bound_size or coords[0] > self.length-self.bound_size:
            raise ValueError("Bad latitude and longitude")

        hour = np.floor(t).astype(int)

        if hour < 10:
            hour = "0" + str(hour)

        DATA = self.GATHERED_ABI_FILES.get(f'{yy}-{ddn}-{hour}')
        print("DATA", DATA)
        if DATA is None:
            DATA = self.gather_files(
                str(yy), str(ddn), str(hour), self.abidata_path)
            self.GATHERED_ABI_FILES[f'{yy}-{ddn}-{hour}'] = DATA

        if DATA["everyten"]:
            minutes = np.round((t - np.floor(t)) * 6).astype(int) * 10
        else:
            minutes = np.round((t - np.floor(t)) * 4).astype(int) * 15

        if minutes == 60:
            if hour != 23:
                hour += 1
                minutes = 0
            else:
                if DATA["everyten"]:
                    minutes = 50
                else:
                    minutes = 45

        minutes = str(minutes)
        if minutes == "0":
            minutes = "00"

        ABI = self.COLLECTED_ABI_DATA.get(f'{yy}-{ddn}-{hour}-{minutes}')
        if ABI is None:
            ABI = self.get_L1B_L2(
                DATA[minutes], DATA["L200"],
                DATA["YYYY"], DATA["DDD"],
                DATA["HH"], self.abidata_path)
            self.COLLECTED_ABI_DATA[f'{yy}-{ddn}-{hour}-{minutes}'] = ABI

        chip = ABI[coords[0]-64:coords[0]+64, coords[1]-64:coords[1]+64, :]

        return chip, coords

    def processFile(self, yy, ddn, orbit, latb):

        cs_file = glob.glob(
            os.path.join(
                self.cloudsat_path,
                '2B-CLDCLASS-LIDAR',
                yy,
                ddn,
                f'{yy}{ddn}*{orbit}*2B-CLDCLASS-LIDAR*P1_R05*.hdf'
            )
        )
        print(cs_file)
        aux_file = glob.glob(
            os.path.join(
                self.cloudsat_path,
                'ECMWF-AUX',
                yy,
                ddn,
                f'{yy}{ddn}*{orbit}*ECMWF-AUX*P1_R05*.hdf'
            )
        )
        print(aux_file)

        if len(cs_file) == 0 or len(aux_file) == 0:
            return

        [Pressure, DEM_elevation, Temperature, Specific_humidity, EC_height, Temperature_2m,
            U10_velocity, V10_velocity, UTC_Time] = self.read_cs_ecmwf(aux_file[0], latbin=latb)

        # ========== read 2b-cldclass-lidar ================================
        [cs_clb, cs_clt, cs_cltype, cs_QC, Latitude,
            Longitude] = self.read_2b_cldclass_lidar(cs_file[0], latbin=latb)
        Longitude[Longitude < 0] += 360

        Pressure = self.interpArray(Pressure, EC_height)
        Temperature = self.interpArray(Temperature, EC_height)
        Specific_humidity = self.interpArray(Specific_humidity, EC_height)

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

        # Find corresponding ABI file to the UTC_Time

        i = 0

        while i < N:
            # chip, coords = self.processTime(
            #    UTC_Time[i], yy, ddn, Latitude[i], Longitude[i])
            
            try:
                chip, coords = self.processTime(
                    UTC_Time[i], yy, ddn, Latitude[i], Longitude[i])
            except ValueError:
                i += 20
                continue
            except ImportError:
                i += 90
                continue
            except Exception as e:
                i += 20
                continue
            
            if i + 46 >= N:
                break

            # dRange = np.arange(i-45, i+46)
            # Reverse the CloudSAT Data
            dRange = np.arange(i+46, i-45, -1)
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
                "Cloud_mask_binary": (
                    cloud_class_mask_40_level[dRange] != 0).astype(int),
                "Cloud_class": cs_cltype[dRange],
            }

            fileName = f'{yy}-{ddn}-{orbit}_{coords[0]}-{coords[1]}-{i}'

            # chip = chip[..., translation]

            np.savez(
                os.path.join(self.output_dir, fileName),
                chip=chip, data=aux_data
            )
            print(fileName, UTC_Time[i])
            i += 45

        return
