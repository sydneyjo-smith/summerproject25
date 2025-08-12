import cv2
import os

img_path = 'data/raw_images/T0018_2019-06-10_BL (6).JPG'
img = cv2.imread(img_path)

print(f"Image dimensions: {img.shape}")  # Output: (height, width, channels)