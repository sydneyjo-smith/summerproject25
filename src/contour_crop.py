import numpy as np
import cv2 as cv
import os

def contour_crop_eye(filename, input_folder='data/raw_images', output_size=(224, 224), padding=30):
    """
    Detects the largest contour in the image (assumed to be the eye region), crops around it, and resizes.

    Parameters:
        filename (str): Name of the image file (e.g. 'image01.jpg').
        input_folder (str): Folder where input images are stored.
        output_size (tuple): Size to resize the final cropped image to (default = 224x224).
        padding (int): Number of pixels to pad around the detected bounding box.

    Returns:
        final_img (np.ndarray): Cropped and resized image.
        filename (str): Original filename (for saving/logging downstream).
    """
    image_path = os.path.join(input_folder, filename)
    img = cv.imread(image_path)

    assert img is not None, f"Image not found or unreadable: {image_path}"

    # Convert to grayscale if it's color
    if len(img.shape) == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Apply Otsu's threshold to binarize the image
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Invert the image if needed (eye may be lighter than background)
    if np.mean(thresh) > 127:
        thresh = cv.bitwise_not(thresh)

    # Find contours
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError(f"No contours found in image: {filename}")

    # Get the largest contour (by area)
    largest_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(largest_contour)

    # Apply padding, making sure not to go out of bounds
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, img.shape[1])
    y2 = min(y + h + padding, img.shape[0])

    cropped = img[y1:y2, x1:x2]
    final_img = cv.resize(cropped, output_size)

    return final_img, filename

#FOR THE TEST ONLY: DELETE ONCE PIPELINE IS ESTABLISHED
if __name__ == "__main__":
    test_filename = "T0004-04-06-2019_BL (1).JPG"
    input_folder = "data/raw_images"

    img, filename = contour_crop_eye(test_filename, input_folder=input_folder)

    # Display the image
    import cv2
    cv2.imshow("Contour Transformation Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()