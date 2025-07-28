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

        self.emissive_mins = np.array(
            [117.04327, 152.00592, 157.96591, 176.15349,
             210.60493, 210.52264, 218.10147, 225.9894],
            dtype=np.float32)

        self.emissive_maxs = np.array(
            [221.07022, 224.44113, 242.3326, 307.42004,
             290.8879, 343.72617, 345.72894, 323.5239],
            dtype=np.float32)

    def __call__(self, img):

        # Reflectance % to reflectance units
        img[:, :, self.reflectance_indices] = \
            img[:, :, self.reflectance_indices] * 0.01

        # Brightness temp scaled to (0,1) range
        img[:, :, self.emissive_indices] = \
            (img[:, :, self.emissive_indices] - self.emissive_mins) / \
            (self.emissive_maxs - self.emissive_mins)

        return img

# -----------------------------------------------------------------------------
# vis_calibrate
# -----------------------------------------------------------------------------
def vis_calibrate(data):
    """Calibrate visible channels to reflectance."""
    solar_irradiance = np.array(2017)
    esd = np.array(0.99)
    factor = np.pi * esd * esd / solar_irradiance

    return data * np.float32(factor) * 100


# -----------------------------------------------------------------------------
# ir_calibrate
# -----------------------------------------------------------------------------
def ir_calibrate(data):
    """Calibrate IR channels to BT."""
    fk1 = np.array(13432.1),
    fk2 = np.array(1497.61),
    bc1 = np.array(0.09102),
    bc2 = np.array(0.99971),

    # if self.clip_negative_radiances:
    #     min_rad = self._get_minimum_radiance(data)
    #     data = data.clip(min=data.dtype.type(min_rad))

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

    def __call__(self, img):

        # Reflectance % to reflectance units
        img[:, :, self.reflectance_indices] = \
            vis_calibrate(img[:, :, self.reflectance_indices])

        # Brightness temp scaled to (0,1) range
        img[:, :, self.emissive_indices] = ir_calibrate(
            img[:, :, self.emissive_indices])

        return img