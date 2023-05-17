import os
import shutil

source_folder = "/home/nikodem/IVSPA/peace_ready"
destination_folder = "/home/nikodem/IVSPA/peace_valid"
files_to_move = 135

# Get a list of files in the source folder
file_list = os.listdir(source_folder)

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Move files from source to destination
moved_count = 0
for filename in file_list:
    if moved_count >= files_to_move:
        break
    
    # Build the file paths
    source_path = os.path.join(source_folder, filename)
    destination_path = os.path.join(destination_folder, filename)

    # Move the file
    shutil.move(source_path, destination_path)
    moved_count += 1

print(f"Moved {moved_count} files from {source_folder} to {destination_folder}.")
