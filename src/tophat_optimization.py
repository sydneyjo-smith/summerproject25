import numpy as np
import cv2
import os

def tophat_enhance_color(filename, raw_folder='data/raw_images', kernel_size=(15, 15)):
    """
    Applies top-hat morphological transformation to color images by converting to LAB color space,
    applying the top-hat operation to the L (lightness) channel, and converting back to BGR.
    This enhances small bright features (e.g., deposits, scars) while preserving color.

    Parameters:
        filename (str): Name of the image file (e.g., 'image01.jpg').
        raw_folder (str): Folder where input images are stored.
        kernel_size (tuple): Size of the structuring element for top-hat (default = (15, 15)).

    Returns:
        result (np.ndarray): Image enhanced with top-hat transform, in BGR format.
        filename (str): Original filename (for saving or tracking).
    """
    image_path = os.path.join(raw_folder, filename)
    with open(image_path, 'rb') as f:
        bgr = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    assert bgr is not None, f"Image not found: {image_path}"

    # Convert to LAB and split channels
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply top-hat transformation to the L channel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    l_tophat = cv2.morphologyEx(l, cv2.MORPH_TOPHAT, kernel)

    # Enhance contrast slightly by stretching intensity range
    l_enhanced = cv2.normalize(l_tophat, None, 0, 255, cv2.NORM_MINMAX)

    # Merge enhanced L with original A and B channels
    lab_tophat = cv2.merge((l_enhanced, a, b))
    result = cv2.cvtColor(lab_tophat, cv2.COLOR_LAB2BGR)

    return result, filename


# FOR TESTING ONLY: DELETE ONCE PIPELINE IS ESTABLISHED
if __name__ == "__main__":
    test_filename = "T0004-04-06-2019_BL (1).JPG"
    raw_folder = "data/raw_images"  # Match parameter name exactly

    img, filename = tophat_enhance_color(test_filename, raw_folder=raw_folder)

    # Display the image
    cv2.imshow("Top-Hat Transformation Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
