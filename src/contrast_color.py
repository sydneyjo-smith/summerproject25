import numpy as np
import cv2
import os

def clahe_preserve_color(image_or_path, raw_folder='data/raw_images', clipLimit=2.0, tileGridSize=(8, 8), fname=None):
    """
    Applies CLAHE contrast enhancement to color images by converting to LAB color space,
    applying CLAHE to the L (lightness) channel, and converting back to BGR.
    This enhances contrast while preserving the original color appearance.

    Parameters:
        image_or_path (str or np.ndarray): 
            - String: filename or full path to the image.
            - np.ndarray: already-loaded image in BGR format.
        raw_folder (str): Folder where raw images are stored (if filename is given).
        clipLimit (float): CLAHE contrast limiting threshold (default = 2.0).
        tileGridSize (tuple): CLAHE grid size (default = (8, 8)).
        fname (str): Optional filename for logging/saving if passing in an array.

    Returns:
        result (np.ndarray): Contrast-enhanced image in BGR format.
        base_filename (str): Base filename only (no directories).
    """
    # Handle if input is an array
    if isinstance(image_or_path, np.ndarray):
        bgr = image_or_path
        base_filename = os.path.basename(fname) if fname else "image.jpg"
    else:
        # It's a path or filename
        if os.path.sep in image_or_path or os.path.isabs(image_or_path):
            image_path = image_or_path
            base_filename = os.path.basename(image_or_path)
        else:
            image_path = os.path.join(raw_folder, image_or_path)
            base_filename = image_or_path  # already base name

        with open(image_path, 'rb') as f:
            bgr = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Image not found: {image_path}")

    # Convert to LAB and split channels
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the lightness channel
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    l_clahe = clahe.apply(l)

    # Merge enhanced lightness with original a & b channels
    lab_clahe = cv2.merge((l_clahe, a, b))
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return result, base_filename
