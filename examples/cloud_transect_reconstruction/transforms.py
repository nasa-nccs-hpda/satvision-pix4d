import numpy as np

class MinMaxEmissiveScaleReflectance(object):
    """
    Performs scaling of MODIS TOA data
    - Scales reflectance percentages to reflectance units (% -> (0,1))
    - Performs per-channel minmax scaling for emissive bands (k -> (0,1))
    """

    def __init__(self):

        self.reflectance_indices = [0, 1, 2, 3, 4, 6]
        self.emissive_indices = [5, 7, 8, 9, 10, 11, 12, 13]

        # self.emissive_mins = np.array(
        #     [117.04327, 152.00592, 157.96591, 176.15349,
        #      210.60493, 210.52264, 218.10147, 225.9894],
        #     dtype=np.float32)

        # self.emissive_maxs = np.array(
        #     [228.68851, 224.83987, 242.3326, 307.42004,
        #      290.8879, 344.21173, 345.72894, 323.5239],
        #     dtype=np.float32)

        self.emissive_mins = np.array(
            [223.122, 178.9174, 204.3739, 204.7677,
             194.8686, 202.1759, 201.3823, 203.3537],
            dtype=np.float32)
        
        self.emissive_maxs = np.array(
            [352.7182, 261.2920, 282.5529, 319.0373,
             295.0209, 324.0677, 321.5254, 285.9848],
            dtype=np.float32)

    def __call__(self, img):

        # Reflectance % to reflectance units
        img[:, :, self.reflectance_indices] = \
            img[:, :, self.reflectance_indices]

        # Brightness temp scaled to (0,1) range
        img[:, :, self.emissive_indices] = \
            (img[:, :, self.emissive_indices] - self.emissive_mins) / \
            (self.emissive_maxs - self.emissive_mins)
        
        # Clip values
        img = np.clip(img, 0.0, 1.0)

        return img

# -----------------------------------------------------------------------------
# vis_calibrate
# -----------------------------------------------------------------------------
def vis_calibrate(data, esun):
    """Calibrate visible channels to reflectance."""
    solar_irradiance = esun
    esd = np.array(0.99)
    factor = np.pi * esd * esd / solar_irradiance

    return data * np.float32(factor)


# -----------------------------------------------------------------------------
# ir_calibrate
# -----------------------------------------------------------------------------
def ir_calibrate(data, fk1, fk2, bc1, bc2):
    """Calibrate IR channels to BT."""

    res = (fk2 / np.log(fk1 / data + 1) - bc1) / bc2
    return res


# -----------------------------------------------------------------------------
# ConvertABIToReflectanceBT
# -----------------------------------------------------------------------------
class ConvertABIToReflectanceBT(object):
    """
    Performs scaling of MODIS TOA data
    - Scales reflectance percentages to reflectance units (% -> (0,1))
    - Performs per-channel minmax scaling for emissive bands (k -> (0,1))
    """

    def __init__(self):

        self.reflectance_indices = [0, 1, 2, 3, 4, 6]
        self.emissive_indices = [5, 7, 8, 9, 10, 11, 12, 13]

        self.esun_values = np.array([
            1631.3351, 957.0699, 2017.1648, 242.54037, 76.8999, 360.90176],
            dtype=np.float32)

        self.fk1_values = np.array([
            202263.0, 35828.3, 30174.0, 19779.9,
            13432.1, 8510.22, 6454.62, 5101.27],
            dtype=np.float32)

        self.fk2_values = np.array([
            3698.19, 2076.95, 1961.38, 1703.83,
            1497.61, 1286.67, 1173.03, 1084.53],
            dtype=np.float32)
        
        self.bc1_values = np.array([
            0.43361, 0.34428, 0.05651, 0.18733,
            0.09102, 0.22516, 0.21702, 0.06266],
            dtype=np.float32)
        
        self.bc2_values = np.array([
            0.99939, 0.99918, 0.99986, 0.99948,
            0.99971, 0.9992, 0.99916, 0.99974],
            dtype=np.float32)

    def __call__(self, img):


        # Reflectance % to reflectance units
        for i, idx in enumerate(self.reflectance_indices):
            img[:, :, idx] = vis_calibrate(img[:, :, idx], self.esun_values[i])

        # Brightness temp scaled to (0,1) range
        for i, idx in enumerate(self.emissive_indices):
            img[:, :, idx] = ir_calibrate(img[:, :, idx], self.fk1_values[i], self.fk2_values[i], self.bc1_values[i], self.bc2_values[i])

        return img