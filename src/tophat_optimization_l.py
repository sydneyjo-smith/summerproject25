import os
import cv2
import numpy as np

def tophat_extract_l_channel(
    img_or_filename,
    fname=None,
    kernel_size=(5, 5)
):
    """
    Applies a top-hat (white top-hat) morphological transformation to the L (lightness)
    channel of an LAB-converted image, then merges it back with the original A & B channels
    to preserve full color information.

    Steps:
       1) Read the image from either a bare filename (looked up in `raw_folder`) or a full path.
       2) Convert BGR → LAB and extract the L channel (perceived luminance).
       3) Build a rectangular structuring element of size `kernel_size`.
       4) Compute white top-hat: L_tophat = L - (L ⊝ kernel) ⊕ kernel
          (i.e., original minus its morphological opening) to emphasize small bright details.
       5) Normalize the result to 8-bit [0, 255]
       6) Merge the enhanced L channel with A and B channels to retain color data
       7) Convert from LAB to RGB color space for downstream processing

    Accepts either:
        - NumPy BGR image array (pass `fname` for logging/saving)
        - Filename or path (loads from disk)

    Parameters:
        img_or_filename: np.ndarray or str
            BGR image array or image filename/path
        fname: str | None
            Required if passing an array; used for saving/logging
        kernel_size: tuple[int, int]
            Structuring element size for morphological opening

    Returns:
        result (np.ndarray): Color (BGR) image with enhanced L channel
        base_filename (str): Filename only, for downstream saving/logging
    """

    # Case 1: already an image array
    if isinstance(img_or_filename, np.ndarray):
        if fname is None:
            raise ValueError("fname must be provided when passing an image array")
        img = img_or_filename
        base_filename = os.path.basename(fname)

    # Case 2: filename/path
    else:
        if os.path.sep in img_or_filename or os.path.isabs(img_or_filename):
            image_path = img_or_filename
            base_filename = os.path.basename(img_or_filename)
        else:
            image_path = os.path.join('data/raw_images', img_or_filename)
            base_filename = img_or_filename

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Image not found or unreadable: {image_path}")

    # Convert to LAB and split channels
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply top-hat to L channel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    top_hat = cv2.subtract(l, cv2.morphologyEx(l, cv2.MORPH_OPEN, kernel))

    # Normalize to [0, 255]
    top_hat_norm = cv2.normalize(top_hat, None, 0, 255, cv2.NORM_MINMAX)

    # Merge enhanced L with original A and B
    lab_merged = cv2.merge((top_hat_norm, a, b))

    # Convert back to BGR
    result = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)

    return result, base_filename

