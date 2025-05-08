import numpy as np
import scipy.io as scio
import h5py

from externals.NTIRE2022_spectral.NTIRE2022Util import load_rgb_filter, make_spectral_bands, resampleHSPicked, projectCube, addNoise
from externals.colour_demosaicing.bayer import mosaicing_CFA_Bayer, demosaicing_CFA_Bayer_Menon2007

TYPICAL_SCENE_REFLECTIVITY = 0.18
NOISE = 750
MAX_VAL_12_BIT = (2 ** 12 - 1)
MAX_VAL_8_BIT = (2 ** 8 - 1)

def make_spectral_filters(cube_bands, qes, qe_bands, interp_mode='linear'):
    """
    Project a spectral array

    :param cube: Input hyperspectral cube
    :param cube_bands: bands of hyperspectral cube
    :param qes: filter response to use for projection
    :param qe_bands: bands of filter response
    :param clipNegative: clip values below 0
    :param interp_mode: interpolation mode for missing values
    :return:
    :return: numpy array of projected data, shape [..., num_channels ]
    """
    if not np.array_equal(qe_bands, cube_bands):  # then sample the qes on the data bands
        dx_qes = qe_bands[1] - qe_bands[0]
        dx_hs = cube_bands[1] - cube_bands[0]
        if np.any(np.diff(qe_bands) != dx_qes) or np.any(np.diff(cube_bands) != dx_hs):
            raise ValueError(f'V81Filter.projectHS - can only interpolate from uniformly sampled bands\n'
                             f'got hs bands: {cube_bands}\n'
                             f'filter bands: {qe_bands}')

        if dx_qes < 0:
            # we assume the qe_bands are sorted ascending inside resampleHSPicked, reverse them
            qes = qes[::-1]
            qe_bands = qe_bands[::-1]

        # find the limits of the interpolation, WE DON'T WANT TO EXTRAPOLATE!
        # the limits must be defined by the data bands so the interpolated qe matches
        min_band = cube_bands[
            np.argwhere(cube_bands >= qe_bands.min()).min()]  # the first data band which has a respective qe value
        max_band = cube_bands[
            np.argwhere(cube_bands <= qe_bands.max()).max()]  # the last data band which has a respective qe value
        # TODO is there a minimal overlap we want to enforce?

        shared_bands = make_spectral_bands(min_band, max_band,
                                           dx_hs)  # shared domain with the spectral resolution of the spectral data
        qes = resampleHSPicked(qes.T, bands=qe_bands, newBands=shared_bands, interpMode=interp_mode,
                               fill_value=np.nan).T

    return qes

def createNoisyRGB(cube, cube_bands, rgb_filter, filter_bands, npe, pattern="RGGB"):
    """
    Generate a noisy RGB image from a spectral cube
    :param cube: Source spectral cube
    :param cube_bands: Bands of spectral cube
    :param rgb_filter: RGB filter for projection
    :param filter_bands: Bands of RGB filter
    :param npe: Noise parameters - defined in "number of photo-electrons"
    :return: RGB image
    """
    qes = make_spectral_filters(cube_bands, rgb_filter, filter_bands)
    rgb_ideal = projectCube(cube, qes, clipNegative=True)
    rgb_raw = mosaicing_CFA_Bayer(rgb_ideal, pattern) 
    noisy_raw = addNoise(rgb_raw, npe=npe) 
    real_raw = ((noisy_raw/noisy_raw.max()) * MAX_VAL_12_BIT).astype(np.uint16)
    
    real_raw = real_raw.astype(np.float64) / MAX_VAL_12_BIT
    noisy_rgb = demosaicing_CFA_Bayer_Menon2007(real_raw, pattern)
    k_EC = (TYPICAL_SCENE_REFLECTIVITY / noisy_rgb.mean()) # Exposure Compensation
    return noisy_rgb*k_EC, qes * k_EC / noisy_raw.max()

if __name__ == "__main__":
    import os
    dataset_dir = 'D:/zwq/Desktop/论文/2_datasets/ARAD_1K/Valid_spectral/'
    target_dir = 'ARAD_1K_demosaic'
    step, shear_dim = 2, 1

    H, W, C = 482, 512, 31
    rgb_filter, filter_bands = load_rgb_filter("./externals/NTIRE2022_spectral/resources/RGB_Camera_QE.csv")
    files =  os.listdir(dataset_dir)
    for file in files:
        if not file.endswith('.mat'):
            continue
        with h5py.File(dataset_dir + file, 'r') as f:
            cube = np.array(f['cube']).transpose(2, 1, 0)
            bands = np.array(f['bands'])
        Y_rgb, A = createNoisyRGB(cube, bands, rgb_filter, filter_bands, 0)
        scio.savemat("ARAD_1K_demosaic/" + file, {"truth":cube, "rgb": Y_rgb, "A": A, "bands": bands})
        print(f"Processing {file} done.")
        
    