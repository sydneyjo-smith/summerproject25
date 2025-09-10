import os
import shutil

# Define your source and destination folders
source_folder = 'data/processed_images'  # or adjust based on your actual path
destination_folder = os.path.expanduser('~/Desktop/Pipeline_Images/PipelineTest3')  # saves to Desktop

# Make destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Loop through and copy files
count = 0
for filename in os.listdir(source_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(destination_folder, filename)
        shutil.move(src_path, dst_path)  # use shutil.move() if you want to move instead of copy
        count += 1

print(f"{count} image files copied to {destination_folder}")
