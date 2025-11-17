import os
from tqdm import tqdm

# Import necessary constants from similar_images.py
from similar_images import SCRIPTDIR, TMP_PATH, LABEL_FILE_EXT, SIMILARS_FILE

def remove_similar_images():
  """
  Reads the similar.txt file and removes the listed images and their corresponding
  label files from the temporary dataset directory.
  """
  similar_images_to_remove = []
  try:
    with open(SIMILARS_FILE, 'r') as f:
      for line in f:
        line = line.strip()
        if not line.startswith('- '): # These are the images to remove
          # Expected format: "  <image_path>::<reason>"
          parts = line.split('::')
          if len(parts) > 0:
            image_path = parts[0].strip()
            similar_images_to_remove.append(image_path)
  except FileNotFoundError:
    print(f"Error: {SIMILARS_FILE} not found. Please run similar_images.py first.")
    return

  if not similar_images_to_remove:
    print("No similar images found to remove.")
    return

  tmp_labels_path = os.path.join(TMP_PATH, 'labels')

  print(f"Found {len(similar_images_to_remove)} images to remove.")

  for image_path in tqdm(similar_images_to_remove, desc="Removing similar images and their annotations..."):
    try:
      if os.path.exists(image_path):
        os.remove(image_path)
      else:
        print(f"Warning: Image not found at {image_path}, skipping removal.")

      label_filename = os.path.splitext(os.path.basename(image_path))[0] + LABEL_FILE_EXT
      label_path = os.path.join(tmp_labels_path, label_filename)
      if os.path.exists(label_path):
        os.remove(label_path)
      else:
        print(f"Warning: Label file not found for {image_path} at {label_path}, skipping removal.")

    except OSError as e:
      print(f"Error removing {image_path} or its label: {e}")

  print("Removal process complete.")

if __name__ == '__main__':
  remove_similar_images()
