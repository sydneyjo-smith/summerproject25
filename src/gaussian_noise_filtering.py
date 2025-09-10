def gaussian_denoise(
    img_or_filename,
    fname=None,
    kernel_size=(5, 5),
    sigma=0,
):
    """
    Applies Gaussian blur for denoising an image.

    Parameters:
        img_or_filename (np.ndarray | str):
            Either an image array (BGR) or a filename/path.
        fname (str | None):
            Base filename for downstream saving (only needed if first arg is an array).
        kernel_size (tuple[int, int]):
            Gaussian kernel size (both odd, positive).
        sigma (int | float):
            Standard deviation in X and Y. If 0, OpenCV infers it from kernel_size.

    Returns:
        denoised_img (np.ndarray): Blurred (denoised) image in BGR.
        base_filename (str): Base filename (no directory), for downstream saving.
    """
    import os
    import cv2
    import numpy as np

    # Case 1: Input is already an image array
    if isinstance(img_or_filename, np.ndarray):
        if fname is None:
            raise ValueError("fname must be provided when passing an image array")
        img = img_or_filename
        base_filename = fname

    # Case 2: Input is a filename/path
    else:
        img_path = img_or_filename
        base_filename = os.path.basename(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Image not found or unreadable: {img_path}")

    # Validate kernel
    if (not isinstance(kernel_size, (tuple, list)) or
        len(kernel_size) != 2 or
        any((k <= 0 or k % 2 == 0) for k in kernel_size)):
        raise ValueError(f"kernel_size must be two positive odd ints, got {kernel_size}")

    # Apply Gaussian blur
    denoised_img = cv2.GaussianBlur(img, tuple(kernel_size), sigma)

    return denoised_img, base_filename



