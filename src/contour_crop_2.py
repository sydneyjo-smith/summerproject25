import os
import cv2
import numpy as np

def contour_crop_binary(
    binary_img,
    original_img_or_path,
    fname="image.jpg",
    output_size=(600,600),
    padding=30
):
    """
    Finds the largest contour in a binary image, crops from the original image (path or array),
    and resizes.

    Parameters:
        binary_img (np.ndarray): Binary (thresholded) image for contour detection.
        original_img_or_path (np.ndarray or str): Either the original color image array OR a file path.
        fname (str): Filename for logging/debugging. Can be base filename or full path.
        output_size (tuple): Final image size after resizing (default: 224x224).
        padding (int): Padding (pixels) around the bounding box.

    Returns:
        final_img (np.ndarray): Cropped and resized image from original.
        base_fname (str): Base filename only, with directory parts removed.
    """

    # Ensure binary image is 2D
    assert binary_img.ndim == 2, f"Binary image must be 2D, got shape: {binary_img.shape}"

    # Load original image if a path is provided
    if isinstance(original_img_or_path, str):
        if os.path.sep in original_img_or_path or os.path.isabs(original_img_or_path):
            image_path = original_img_or_path
        else:
            image_path = os.path.join('data/raw_images', original_img_or_path)
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise ValueError(f"Image not found or unreadable: {image_path}")
        base_fname = os.path.basename(original_img_or_path)
    else:
        # Already an ndarray
        original_img = original_img_or_path
        base_fname = os.path.basename(fname)

    # Check size match
    assert original_img.shape[:2] == binary_img.shape, \
        f"Image size mismatch between original ({original_img.shape}) and binary mask ({binary_img.shape})"

    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No contours found in binary mask for: {base_fname}")

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Filter for plausible aspect ratio
    aspect_ratio = w / h
    if aspect_ratio < 0.6 or aspect_ratio > 2.5:
        raise ValueError(f"Aspect ratio {aspect_ratio:.2f} out of range for image: {base_fname}")

    # Apply padding
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, original_img.shape[1])
    y2 = min(y + h + padding, original_img.shape[0])

    # Crop and resize
    cropped = original_img[y1:y2, x1:x2]
    final_img = cv2.resize(cropped, output_size)

    # Ensure the image is returned to 3 channels (BGR) for AI integration
    if len(final_img.shape) == 2:  # aka grayscale
        final_img = cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)


    return final_img, base_fname

