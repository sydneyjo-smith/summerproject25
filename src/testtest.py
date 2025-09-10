import os
import cv2
import numpy as np
from homomorphic_filter import homomorphic_filter_color  # adjust path if needed
from contour_crop import contour_crop_eye

def process_image_with_hf_and_contour(filename, input_folder='data/raw_images', output_folder='data/hf_contour_output'):
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Apply Homomorphic Filtering
    try:
        hf_img, _ = homomorphic_filter_color(filename, fname=filename)
    except Exception as e:
        print(f"[FAIL] {filename}: Error during homomorphic filtering → {e}")
        return

    # Step 2: Attempt to crop eye using contour detection
    try:
        cropped_img, base_filename = contour_crop_eye(
            img_or_filename=hf_img,
            fname=filename,
            input_folder=input_folder,
            output_size=(600, 600),  # Required final shape for AI integration 
            padding=30
        )
    except Exception as e:
        print(f"[FAIL] {filename}: Contour crop failed → {e}")
        return

    # Step 3: Final sanity checks
    if cropped_img is None or cropped_img.shape[:2] != (600, 600):
        print(f"[FAIL] {filename}: Cropping or resizing failed (final shape {cropped_img.shape})")
        return

    gray_check = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    if np.mean(gray_check) < 5:
        print(f"[FAIL] {filename}: Image too dark after processing (mean pixel value = {np.mean(gray_check):.2f})")
        return

    # Step 4: Convert to RGB before saving
    rgb_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

    # Step 5: Save final image
    output_path = os.path.join(output_folder, base_filename)
    cv2.imwrite(output_path, rgb_img)
    print(f"[SUCCESS] {filename} processed and saved to {output_path}")