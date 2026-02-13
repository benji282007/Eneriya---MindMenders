import os
from PIL import Image
import imagehash

def process_images(folder_path, threshold=4):
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    hashes = {}
    duplicates = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    # 1. Identify Duplicates
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)])
    
    print(f"Scanning {len(files)} files for similarities...")

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                current_hash = imagehash.dhash(img)
            
            is_duplicate = False
            for existing_hash in hashes:
                # Hamming distance calculation
                if current_hash - existing_hash <= threshold:
                    duplicates.append(file_path)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                hashes[current_hash] = file_path
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")

    # 2. Delete Duplicates
    if not duplicates:
        print("No duplicates found.")
    else:
        for file_path in duplicates:
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")
        print(f"Deleted {len(duplicates)} similar images.")

    # 3. Rename Remaining Files (Two-Pass to avoid FileExistsError)
    remaining_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)])
    
    if not remaining_files:
        print("No files left to rename.")
        return

    print(f"Renaming {len(remaining_files)} files...")
    
    # Pass 1: Rename to temporary unique names
    temp_paths = []
    for filename in remaining_files:
        old_path = os.path.join(folder_path, filename)
        temp_path = os.path.join(folder_path, f"TEMP_RENAME_{filename}")
        os.rename(old_path, temp_path)
        temp_paths.append(temp_path)

    # Pass 2: Rename from temporary to final format
    for index, temp_path in enumerate(temp_paths, start=1):
        new_name = f"frame_{str(index).zfill(4)}.jpg"
        final_path = os.path.join(folder_path, new_name)
        os.rename(temp_path, final_path)

    print("Cleanup and renaming complete.")