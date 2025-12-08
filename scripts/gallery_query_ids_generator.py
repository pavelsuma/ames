import os
from PIL import Image
from tqdm import tqdm

# ================= CONFIGURATION =================
# Path to your ILIAS dataset root
DATASET_ROOT = "/private/chechik-lab/ofirk/ILIAS/ilias_core" 

# Output file names
QUERY_LIST_FILE = "test_query.txt"
GALLERY_LIST_FILE = "test_gallery.txt"
# =================================================

def get_image_info(file_path):
    """
    Returns (width, height) of an image.
    """
    try:
        with Image.open(file_path) as img:
            return img.size
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    query_lines = []
    gallery_lines = []

    if not os.path.exists(DATASET_ROOT):
        print(f"Error: Dataset path '{DATASET_ROOT}' does not exist.")
        return

    # Get all items in the root directory
    all_items = os.listdir(DATASET_ROOT)
    
    # Filter for directories that end in a number (e.g., 'name_000')
    instance_dirs = []
    for d in all_items:
        full_path = os.path.join(DATASET_ROOT, d)
        if os.path.isdir(full_path):
            # Split by underscore to find the ID at the end
            parts = d.split('_')
            # Check if the last part is a number (the ID)
            if parts and parts[-1].isdigit():
                instance_dirs.append(d)

    # Sort them so 000 comes before 001
    instance_dirs.sort(key=lambda x: int(x.split('_')[-1]))

    print(f"Found {len(instance_dirs)} valid instances (ending in _XXX). Processing...")

    for instance_folder in tqdm(instance_dirs):
        # Extract ID from the end of the string
        class_id = int(instance_folder.split('_')[-1])
        
        instance_full_path = os.path.join(DATASET_ROOT, instance_folder)

        # ---------------- PROCESS QUERY ----------------
        # Look for Q_XXX_00.jpg inside /query
        query_path = os.path.join(instance_full_path, "query")
        if os.path.exists(query_path):
            for file_name in os.listdir(query_path):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_file_path = os.path.join(query_path, file_name)
                    dims = get_image_info(full_file_path)
                    
                    if dims:
                        # Relpath ensures format: specific_name_000/query/Q_001_00.jpg
                        rel_path = os.path.relpath(full_file_path, DATASET_ROOT)
                        line = f"{rel_path},{class_id},{dims[0]},{dims[1]}"
                        query_lines.append(line)

        # ---------------- PROCESS GALLERY (POS) ----------------
        # Look for PXXX_YY.jpg inside /pos
        pos_path = os.path.join(instance_full_path, "pos")
        if os.path.exists(pos_path):
            for file_name in os.listdir(pos_path):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_file_path = os.path.join(pos_path, file_name)
                    dims = get_image_info(full_file_path)
                    
                    if dims:
                        rel_path = os.path.relpath(full_file_path, DATASET_ROOT)
                        line = f"{rel_path},{class_id},{dims[0]},{dims[1]}"
                        gallery_lines.append(line)

    # Write outputs
    with open(QUERY_LIST_FILE, "w") as f:
        f.write("\n".join(query_lines))
    
    with open(GALLERY_LIST_FILE, "w") as f:
        f.write("\n".join(gallery_lines))

    print(f"\nDone!")
    print(f"Generated {QUERY_LIST_FILE} with {len(query_lines)} lines.")
    print(f"Generated {GALLERY_LIST_FILE} with {len(gallery_lines)} lines.")

if __name__ == "__main__":
    main()