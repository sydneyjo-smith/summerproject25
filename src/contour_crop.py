import numpy as np
import cv2 as cv
import os

def contour_crop_eye(img_or_filename, fname=None, input_folder='data/raw_images', output_size=(600,600), padding=30):
    """
    Detects the largest contour in the image (assumed to be the eye region), crops around it, and resizes.

    Accepts:
        - NumPy image array (BGR) + fname
        - Filename/path (loads from disk)

    Returns:
        final_img (np.ndarray): Cropped and resized image.
        base_filename (str): Original filename for saving/logging.
    """
    # Case 1: already an image array
    if isinstance(img_or_filename, np.ndarray):
        if fname is None:
            raise ValueError("fname must be provided when passing an image array")
        img = img_or_filename
        base_filename = os.path.basename(fname)

    # Case 2: filename/path
    else:
        base_filename = os.path.basename(img_or_filename)
        image_path = os.path.join(input_folder, base_filename)
        img = cv.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found or unreadable: {image_path}")

    # Convert to grayscale if it's color (needed for the sake of the CV function, will convert back after)
    if len(img.shape) == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Apply Otsu's threshold 
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Invert if needed to help find the contours
    if np.mean(thresh) > 127:
        thresh = cv.bitwise_not(thresh)

    # Find contours
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No contours found in image: {base_filename}")

    # Get largest contour (most likely the iris)
    largest_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(largest_contour)

    # Apply padding with bounds check
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, img.shape[1])
    y2 = min(y + h + padding, img.shape[0])

    # Create the variable for the final image
    cropped = img[y1:y2, x1:x2]
    final_img = cv.resize(cropped, output_size)

    # Ensure the image is returned to 3 channels (BGR) for AI integration
    if len(final_img.shape) == 2:  # This would mean the image is in graydidscale
        final_img = cv.cvtColor(final_img, cv.COLOR_GRAY2BGR)

    return final_img, base_filename
