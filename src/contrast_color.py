import numpy as np
import cv2 
import os

def clahe_preserve_color(filename, raw_folder='data/raw_images', clipLimit=2.0, tileGridSize=(8, 8)):
    """
    Applies CLAHE contrast enhancement to color images by converting to LAB color space,
    applying CLAHE to the L (lightness) channel, and converting back to BGR.
    This enhances contrast while preserving the original color appearance.

    Parameters:
        filename (str): Name of the image file (e.g., 'image01.jpg').
        raw_folder (str): Folder path where raw images are stored.
        clipLimit (float): CLAHE contrast limiting threshold (default = 2.0).
        tileGridSize (tuple): CLAHE grid size (default = (8, 8)).

    Returns:
        result (np.ndarray): Contrast-enhanced image in BGR format.
        filename (str): Original filename (for saving or tracking).
    """
    image_path = os.path.join(raw_folder, filename)
    bgr = cv2.imread(image_path)
    assert bgr is not None, f"Image not found: {image_path}"

    # Convert to LAB and split channels
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the lightness channel
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    l_clahe = clahe.apply(l)

    # Merge enhanced lightness with original a & b channels
    lab_clahe = cv2.merge((l_clahe, a, b))
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return result, filename

#FOR THE TEST ONLY: DELETE ONCE PIPELINE IS ESTABLISHED
if __name__ == "__main__":
    test_filename = "T0004-04-06-2019_BL (1).JPG"
    raw_folder = "data/raw_images"

    img, filename = clahe_preserve_color(test_filename, raw_folder=raw_folder)

    # Display the image
    import cv2
    cv2.imshow("CLAHE Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()