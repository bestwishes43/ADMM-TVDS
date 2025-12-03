import numpy as np
import scipy.io as scio
import h5py

from externals.NTIRE2022_spectral.NTIRE2022Util import load_rgb_filter, make_spectral_bands, resampleHSPicked, projectCube
from externals.colour_demosaicing.bayer import mosaicing_CFA_Bayer, demosaicing_CFA_Bayer_Menon2007
from skimage.restoration import estimate_sigma
import bm3d

from externals.NTIRE2022_spectral.Conf import TYPICAL_SCENE_REFLECTIVITY,  MAX_VAL_12_BIT, MAX_VAL_8_BIT

BASE_NPE = 400.   # number of photo-electrons
BASE_SIGMA = 8.

NOISE_PARAM_CASE1 = (2, 0.5) # 相对ISO，相对曝光时间
NOISE_PARAM_CASE2 = (1, 1)
NOISE_PARAM_CASE3 = (0.5, 2)

def make_spectral_filters(cube_bands, qes, qe_bands, interp_mode='linear'):
    """
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
                               fill_value=np.nan).T # type: ignore

    return qes

def addNoise(signal, ratio=1.):
    """
    Add camera simulated noise to an Image, based on Poisson and Gaussian Normal dark noise model.

    return: A new Image with added noise (without changing the scale of the signal)
    """
    if BASE_NPE == 0:
        return signal
    ISO = NOISE_PARAM[0]
    exposure_time = NOISE_PARAM[1]
    scale = BASE_NPE * exposure_time * ratio
    shotNoiseSignal = signal.clip(0, None) * scale
    shotNoiseSignal = np.random.poisson(shotNoiseSignal)  # Clip signal to positive values only. Randomize signal by poisson Shot noise model
    shotNoiseSignal = shotNoiseSignal * ISO
    noisy_img = shotNoiseSignal + np.random.normal(0., BASE_SIGMA*ISO, size=shotNoiseSignal.shape)
    noisy_img = noisy_img / scale / ISO  # Total noisy signal. (scaled to original range)
    return noisy_img

def createNoisyRGB(cube, cube_bands, rgb_filter, filter_bands, pattern="RGGB", split_ratio=0.5, ideal=True):
    
    # %% 1. CFA Samping
    # Ideal RGB Projection
    qes = make_spectral_filters(cube_bands, rgb_filter, filter_bands)
    rgb_ideal = projectCube(cube, qes, clipNegative=True)
    if ideal:
        return rgb_ideal, qes, 0.
    # Sampling
    rgb_raw = mosaicing_CFA_Bayer(rgb_ideal, pattern) 
    
    # %% 2. Add Noise
    # Add Camera Noise
    noisy_raw = addNoise(rgb_raw, split_ratio)

    # Automatic Gain Control Based on Global Metering
    # scale_factor = TYPICAL_SCENE_REFLECTIVITY / noisy_raw.mean()
    scale_factor = 1.0 / noisy_raw.max() # 防止过曝
    noisy_raw = noisy_raw * scale_factor
    # Add ADC Noise
    real_raw = ((noisy_raw * MAX_VAL_12_BIT).clip(0, MAX_VAL_12_BIT)).astype(np.uint16)
    # %% Simplified ISP
    real_raw = real_raw.astype(np.float64) / MAX_VAL_12_BIT
    demosaiced_rgb = demosaicing_CFA_Bayer_Menon2007(real_raw, pattern) 
    sigma_est = np.array(estimate_sigma(demosaiced_rgb, channel_axis=-1))
    denoise_rgb = bm3d.bm3d(demosaiced_rgb, sigma_est).clip(0, None) # type: ignore

    RMSE = np.sqrt(np.mean(np.square((demosaiced_rgb - rgb_ideal*scale_factor)))) # 用demosaiced_rgb计算，反映成像性能
    PSNR = 20 * np.log10(1. / RMSE)
    return denoise_rgb/scale_factor, qes, PSNR

def createMosaicRGB(cube, cube_bands, rgb_filter, filter_bands, pattern="RGGB", factor=0.5, ideal=True):
    
    # %% 1. CFA Samping
    # Ideal RGB Projection
    qes = make_spectral_filters(cube_bands, rgb_filter, filter_bands)
    rgb_ideal = projectCube(cube, qes, clipNegative=True)
    
    rgb_raw = mosaicing_CFA_Bayer(rgb_ideal, pattern) 
    
    scale_factor = 1.0 / rgb_raw.max()
    rgb_raw = rgb_raw * scale_factor

    demosaiced_rgb = demosaicing_CFA_Bayer_Menon2007(rgb_raw, pattern) 
    demosaiced_rgb = demosaiced_rgb * (1-factor) + rgb_ideal*scale_factor * factor
    RMSE = np.sqrt(np.mean(np.square((demosaiced_rgb - rgb_ideal*scale_factor)))) # 用demosaiced_rgb计算，反映成像性能
    PSNR = 20 * np.log10(1. / RMSE)
    return demosaiced_rgb/scale_factor, qes, PSNR

def shift(inputs, step:int, dim:int=1):
    """ Apply shear transformation to the input tensor along the specified dimension.
    Args:
        inputs (torch.Tensor): Input tensor of shape [..., H, W, C].
        step (int): Step size for the shear transformation.
        dim (int): Dimension along which to apply the shear transformation.
    Returns:
        torch.Tensor: Sheared tensor of shape [..., H, W+step*(C-1), C].
    """
    nC = inputs.shape[-1]
    n_order = len(inputs.shape)
    
    padsize = []
    for i in range(n_order):
        if i == dim:
            padsize.append((0, (nC - 1) * step))
        else:
            padsize.append((0, 0))

    output = np.pad(inputs, padsize, mode='constant', constant_values=0.0)
    
    for i in range(1, nC):
        slices = [slice(None)] * (n_order - 1) + [i]
        output[tuple(slices)] = np.roll(output[tuple(slices)], step * i, axis=dim)
    return output

def createNoisyCASSI(cube, mask, split_ratio=0.5, ideal=True):
    # %% 1. Ideal SD-CASSI Projection
    cassi_ideal = np.sum(shift(mask*cube, 2, 1), axis=-1)
    if ideal:
        return cassi_ideal
    # %% 2. Add Noise
    # Add Camera Noise
    noisy_raw = addNoise(cassi_ideal, split_ratio)

    # Automatic Gain Control Based on Global Metering
    # scale_factor = TYPICAL_SCENE_REFLECTIVITY/ noisy_raw.mean()
    scale_factor = 1.0 / noisy_raw.max() # 防止过曝
    noisy_raw = noisy_raw * scale_factor
    real_raw = ((noisy_raw*MAX_VAL_12_BIT).clip(0, MAX_VAL_12_BIT)).astype(np.uint16)

    # %% Simplified ISP
    real_raw = real_raw.astype(np.float64) / MAX_VAL_12_BIT

    RMSE = np.sqrt(np.mean(np.square((real_raw - cassi_ideal*scale_factor))))
    PSNR = 20 * np.log10(1. / RMSE)
    return real_raw/scale_factor, PSNR

def load_cassi(mask_path):
    mask_real = scio.loadmat(mask_path)['mask']
    mask_real = np.repeat(mask_real[:, :, None], 31, axis=-1)
    return mask_real

if __name__ == "__main__":
    import os
    import shutil
    dataset_dir = './datasets/ARAD/'
    step, shear_dim = 2, 1

    H, W, C = 482, 512, 31
    mask3D = load_cassi("./datasets/mask.mat")
    rgb_filter, filter_bands = load_rgb_filter("./externals/NTIRE2022_spectral/resources/RGB_Camera_QE.csv")
    files =  os.listdir(dataset_dir)
    nSample = 10

    CASEs = ["Poor", "Medium", "Good"]
    splits = [0.3, 0.5, 0.7]

    target_dir = "datasets/ARAD-simu/"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    shutil.copy("./datasets/mask.mat", target_dir)
    
    noise_levels_of_datasets = np.zeros((nSample, len(CASEs), len(splits), 2))
    for CASE, NOISE_PARAM in zip(CASEs, [NOISE_PARAM_CASE1, NOISE_PARAM_CASE2, NOISE_PARAM_CASE3]):
        if not os.path.exists(target_dir + CASE):
            os.makedirs(target_dir + CASE)
        for split_ratio in splits:
            if not os.path.exists(target_dir + CASE + f"/split{split_ratio}/"):
                os.makedirs(target_dir + CASE + f"/split{split_ratio}/")
            print("Scene | RGB PSNR | CASSI PSNR")
            for i, file in enumerate(files[0:nSample]):
                if not file.endswith('.mat'):
                    continue
                with h5py.File(dataset_dir + file, 'r') as f:
                    norm_factor = np.array(f['norm_factor'])
                    cube = np.array(f['cube']).transpose(2, 1, 0)[113:369, 128:384, :] * norm_factor
                    bands = np.array(f['bands'])
                Y_rgb, _, PSNR_RGB = createNoisyRGB(cube, bands, rgb_filter, filter_bands, split_ratio=1-split_ratio, ideal=False)
                Y_cassi, PSNR_CASSI = createNoisyCASSI(cube, mask3D, split_ratio=split_ratio, ideal=False)
                
                noise_levels_of_datasets[i, CASEs.index(CASE), splits.index(split_ratio), 0] = PSNR_RGB
                noise_levels_of_datasets[i, CASEs.index(CASE), splits.index(split_ratio), 1] = PSNR_CASSI
                
                print(f"{CASE}-{split_ratio:.1f}, Scene{i+1:02d} | {PSNR_RGB:.4f} | {PSNR_CASSI:.4f} ")
                scio.savemat(target_dir + CASE + f"/split{split_ratio}/" + file, {"truth":cube, "rgb": Y_rgb, "cassi":Y_cassi})
    scio.savemat(target_dir + "noise_levels_of_datasets.mat", {"noise_levels":noise_levels_of_datasets})
    
        
    if not os.path.exists(target_dir + "Ideal"):
        os.makedirs(target_dir + "Ideal")
    if not os.path.exists(target_dir + "Ideal/Valid"):
        os.makedirs(target_dir + "Ideal/Valid")
    for i, file in enumerate(files[0:nSample]):
        if not file.endswith('.mat'):
            continue
        with h5py.File(dataset_dir + file, 'r') as f:
            cube = np.array(f['cube']).transpose(2, 1, 0)[113:369, 128:384, :]
            bands = np.array(f['bands'])
        Y_rgb, _, _ = createMosaicRGB(cube, bands, rgb_filter, filter_bands, ideal=True)
        Y_cassi = createNoisyCASSI(cube, mask3D, ideal=True)
        scio.savemat(target_dir + "Ideal/Valid/" + file, {"truth":cube, "rgb": Y_rgb, "cassi":Y_cassi})
