import os
import cv2
import numpy as np

def homomorphic_filter_color(
    img_or_filename,
    fname=None,
    gamma_l=0.5,
    gamma_h=2.0,
    cutoff=30.0
):
    """
    Applies homomorphic filtering to reduce uneven illumination and enhance contrast.
    Operates on the L (lightness) channel of LAB color space to preserve color.

    Accepts either:
        - A NumPy BGR image array (pass `fname` as well for logging/saving)
        - A filename or full path (loads image from disk)

    Returns:
        result (np.ndarray): Processed BGR image
        base_filename (str): Base filename only, for saving/logging
    """
    # Case 1: Input is already an image array
    if isinstance(img_or_filename, np.ndarray):
        if fname is None:
            raise ValueError("fname must be provided when passing an image array")
        bgr = img_or_filename
        base_filename = os.path.basename(fname)

    # Case 2: Input is a filename/path
    else:
        if os.path.sep in img_or_filename or os.path.isabs(img_or_filename):
            image_path = img_or_filename
            base_filename = os.path.basename(img_or_filename)
        else:
            image_path = os.path.join('data/raw_images', img_or_filename)
            base_filename = img_or_filename

        with open(image_path, 'rb') as f:
            bgr = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Image not found: {image_path}")

    # Convert to LAB and split channels
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Normalize L to [0,1] and apply log transform
    l_float = l.astype(np.float32) / 255.0
    log_l = np.log1p(l_float)

    # FFT and shift
    dft = np.fft.fft2(log_l)
    dft_shift = np.fft.fftshift(dft)

    # Gaussian high-pass filter
    rows, cols = l.shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows, dtype=np.float32)
    v = np.arange(cols, dtype=np.float32)
    V, U = np.meshgrid(v - ccol, u - crow)
    D = np.sqrt(U**2 + V**2)
    H = (gamma_h - gamma_l) * (1 - np.exp(-(D**2) / (2 * (cutoff**2)))) + gamma_l

    # Apply filter
    filtered = H * dft_shift

    # Inverse FFT
    dft_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(dft_ishift)
    img_homo = np.real(img_back)

    # Exponentiate and rescale to [0,255]
    exp_l = np.expm1(img_homo)
    exp_l = np.clip(exp_l, 0, 1)
    l_filtered = (exp_l * 255).astype(np.uint8)

    # Merge channels back and convert to BGR
    lab_filtered = cv2.merge((l_filtered, a, b))
    result = cv2.cvtColor(lab_filtered, cv2.COLOR_LAB2BGR)

    return result, base_filename


