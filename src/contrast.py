#i think these are already done- delete once pipeline is finalized
import numpy as np
import cv2 as cv
import os

#color version
def clahe_color_image(filename, raw_folder='data/raw_images', clipLimit=1.5, tileGridSize=(8, 8)):
    """
    Applies CLAHE contrast methods to color images: loads the image in from raw_images folder,
    converts to grayscale, and enhances contrast, then changes the color scale back (this will still appear gray to the human eye).

    Parameters:
        filename (str): Name of the image file (e.g. 'image01.jpg').
        raw_folder (str): Folder path where raw images are stored.
        clipLimit (float): CLAHE contrast limiting threshold (default = 2.0).
        tileGridSize (tuple): CLAHE grid size (default = (8, 8)).

    Returns:
        processed_img (np.ndarray): Contrast-enhanced image in BGR color format.
        filename (str): Original filename (to use when saving later).
    """
    image_path = os.path.join(raw_folder, filename)
    color_img = cv.imread(image_path)

    assert color_img is not None, f"Image not found or unreadable: {image_path}"

    # Convert to grayscale for CLAHE
    gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)

    # Apply CLAHE
    clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl1 = clahe.apply(gray_img)

    # Convert back to BGR format
    processed_img = cv.cvtColor(cl1, cv.COLOR_GRAY2BGR)

    return processed_img, filename

#then it can be sent to the next step in the pipeline!! yay!

#FOR THE TEST ONLY: DELETE ONCE PIPELINE IS ESTABLISHED
if __name__ == "__main__":
    test_filename = "T0004-04-06-2019_BL (1).JPG"
    raw_folder = "data/raw_images"

    img, filename = clahe_color_image(test_filename, raw_folder=raw_folder)

    # Display the image
    import cv2
    cv2.imshow("CLAHE Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

