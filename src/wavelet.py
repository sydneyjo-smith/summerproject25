from skimage import img_as_float, img_as_ubyte, color
from skimage.restoration import denoise_wavelet
import numpy as np
import os
import cv2

def wavelet_denoise_lab_cv(
    image_or_path,
    raw_folder='data/raw_images',
    wavelet='db1',
    method='BayesShrink',
    mode='soft',
    wavelet_levels=2,
    rescale_sigma=True,
    ab_scale=0.9,  # 1.0 = keep color; <1.0 reduces blue cast
    fname=None
):
    """
    Applies wavelet denoising on the L channel of Lab color space to reduce noise
    while preserving color information.

    Parameters:
        image_or_path (str or np.ndarray):
            - str: filename or full path to the image.
            - np.ndarray: already-loaded image (BGR).
        raw_folder (str): Folder for loading if filename is given.
        wavelet, method, mode, wavelet_levels, rescale_sigma: Denoising parameters.
        ab_scale (float): Optional scale factor to reduce a/b channels (color cast).
        fname (str): Optional filename for logging/saving when passing in an array.

    Returns:
        result (np.ndarray): Denoised image in BGR format.
        base_filename (str): Base filename for saving/logging.
    """

    # Handle array input
    if isinstance(image_or_path, np.ndarray):
        bgr = image_or_path
        base_filename = os.path.basename(fname) if fname else "image.jpg"
    else:
        # Handle string path/filename
        if os.path.sep in image_or_path or os.path.isabs(image_or_path):
            img_path = image_or_path
            base_filename = os.path.basename(image_or_path)
        else:
            img_path = os.path.join(raw_folder, image_or_path)
            base_filename = image_or_path

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

    # Convert to RGB float [0,1]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_f = img_as_float(rgb)

    # RGB -> Lab (L in [0,100])
    lab = color.rgb2lab(rgb_f)
    L = lab[..., 0] / 100.0  # Normalize for denoising

    # Wavelet denoise L channel
    Ld = denoise_wavelet(
        L,
        channel_axis=None,
        wavelet=wavelet,
        method=method,
        mode=mode,
        wavelet_levels=wavelet_levels,
        rescale_sigma=rescale_sigma
    )
    lab[..., 0] = np.clip(Ld * 100.0, 0, 100)

    # Optionally desaturate a/b channels
    if ab_scale != 1.0:
        lab[..., 1] *= ab_scale
        lab[..., 2] *= ab_scale

    # Convert back to RGB then BGR
    rgb_out = np.clip(color.lab2rgb(lab), 0, 1)
    bgr_out = cv2.cvtColor(img_as_ubyte(rgb_out), cv2.COLOR_RGB2BGR)

    return bgr_out, base_filename