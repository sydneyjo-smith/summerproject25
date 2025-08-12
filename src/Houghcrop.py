#i think these are already done- delete once pipeline is finalized
import numpy as np
import cv2 as cv
import os

#color or grayscale version (depends on what was passed in!)
def hough_crop_eye(filename, input_folder='data/raw_images', 
                   dp=1.2, minDist=100, param1=100, param2=60, 
                   minRadius=30, maxRadius=150, output_size=(224, 224)):
    """
    Uses Hough Circle Transform to detect circular features (e.g., cornea or pupil) and crops the image around the detected circle.

    Parameters:
        filename (str): Name of the image file (e.g. 'image01.jpg').
        input_folder (str): Folder where preprocessed (CLAHE or grayscale) images are stored.
        dp (float): Inverse ratio of accumulator resolution to image resolution (usually 1.0–2.0).
        minDist (int): Minimum distance between detected circles.
        param1 (int): Higher threshold for Canny edge detector (lower is half this).
        param2 (int): Accumulator threshold for circle detection (lower = more circles).
        minRadius (int): Minimum radius of circles to detect.
        maxRadius (int): Maximum radius of circles to detect.
        output_size (tuple): Final size of the cropped and resized image.

    Returns:
        final_img (np.ndarray): Cropped and resized image centered on detected circle.
        filename (str): Original filename (for saving/logging downstream).
    """
    image_path = os.path.join(input_folder, filename)
    img = cv.imread(image_path)

    assert img is not None, f"Image not found or unreadable: {image_path}"

    # Scale the image down because its WAY too big
    scale_factor = 0.2
    small = cv.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)

    # Convert to grayscale if needed- this will change based on opitmal steps from before!
    if len(img.shape) == 3:
        gray = cv.cvtColor(small, cv.COLOR_BGR2GRAY)
    else:
        gray = small.copy()

    # Use Hough Circle Transform
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                               param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        x, y, r = circles[0]  # Take the first (most confident) circle

        # Define crop box with some padding
        pad = 20
        x1 = max(x - r - pad, 0)
        y1 = max(y - r - pad, 0)
        x2 = min(x + r + pad, img.shape[1])
        y2 = min(y + r + pad, img.shape[0])
        cropped = img[y1:y2, x1:x2]

        # Resize to desired input size for AI model
        final_img = cv.resize(cropped, output_size)
    else:
        raise ValueError(f"Hough Transform failed to detect a circle in: {filename}")

    return final_img, filename

#FOR THE TEST ONLY: DELETE ONCE PIPELINE IS ESTABLISHED
if __name__ == "__main__":
    test_filename = "T0018_2019-06-10_BL (6).JPG"
    input_folder = "data/raw_images"

    img, filename = hough_crop_eye(test_filename, input_folder=input_folder)

    # Display the image
    import cv2
    cv2.imshow("Hough Transformation Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()