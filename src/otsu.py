def otsu_threshold(image, fname='processed_image.jpg'):
    """
    Applies Otsu's thresholding to the provided image and returns the binary mask.

    Parameters:
        image (np.ndarray): Image array in BGR format.
        fname (str): Filename string, used for tracking/saving purposes.
                     Can be a base filename (e.g., "image01.jpg") or a full path.

    Returns:
        thresh (np.ndarray): Binary image after Otsu's thresholding.
        fname (str): Base filename only, with directory parts removed.
    """
    import os
    import cv2 as cv
    import numpy as np

    # Ensure fname is just the base name
    base_fname = os.path.basename(fname)

    # Convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding
    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Optional: Invert if foreground is light
    if np.mean(thresh) > 127:
        thresh = cv.bitwise_not(thresh)

    return thresh, base_fname
