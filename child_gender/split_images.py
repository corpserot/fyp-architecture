import os
import shutil
import random
from typing import List, Tuple, Dict

from tqdm import tqdm

# Constants for dataset paths and split ratios
SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPTDIR, 'dataset_tmp')
OUTPUT_PATH = os.path.join(SCRIPTDIR, 'dataset_raw')

# Split ratios
TRAIN_RATIO = 0.75
VALID_RATIO = 0.15
TEST_RATIO = 0.1

# Ensure ratios sum to 1.0
assert 0.9999 < TRAIN_RATIO + VALID_RATIO + TEST_RATIO < 1.0001, "Split ratios must sum to 1.0"

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
LABEL_FILE_EXT = '.txt'

def get_image_label_paths(
  dataset_path: str,
) -> List[Tuple[str, str]]:
  """
  Collects all image and corresponding label paths from the dataset.

  Args:
    dataset_path (str): The root path of the dataset (e.g., 'dataset_raw').
                        Assumes 'images' and 'labels' subdirectories exist.

  Returns:
    List[Tuple[str, str]]: A list of tuples, where each tuple contains
                           (image_path, label_path).
  """
  images_dir = os.path.join(dataset_path, 'images')
  labels_dir = os.path.join(dataset_path, 'labels')

  if not os.path.exists(images_dir):
    raise FileNotFoundError(f"Image directory not found: {images_dir}")
  if not os.path.exists(labels_dir):
    raise FileNotFoundError(f"Label directory not found: {labels_dir}")

  image_label_paths: List[Tuple[str, str]] = []
  for filename in os.listdir(images_dir):
    if filename.lower().endswith(IMAGE_EXTS):
      image_path = os.path.join(images_dir, filename)
      label_filename = os.path.splitext(filename)[0] + LABEL_FILE_EXT
      label_path = os.path.join(labels_dir, label_filename)

      if os.path.exists(label_path):
        image_label_paths.append((image_path, label_path))
      else:
        print(f"Warning: No corresponding label found for image: {image_path}")
  return image_label_paths

def split_dataset(
  image_label_paths: List[Tuple[str, str]],
  train_ratio: float,
  valid_ratio: float,
  test_ratio: float,
) -> Dict[str, List[Tuple[str, str]]]:
  """
  Splits the list of image and label paths into train, validation, and test sets.

  Args:
    image_label_paths (List[Tuple[str, str]]): List of (image_path, label_path) tuples.
    train_ratio (float): Proportion of data for the training set.
    valid_ratio (float): Proportion of data for the validation set.
    test_ratio (float): Proportion of data for the test set.

  Returns:
    Dict[str, List[Tuple[str, str]]]: A dictionary with 'train', 'valid', and 'test' keys,
                                     each containing a list of (image_path, label_path) tuples.
  """
  random.shuffle(image_label_paths)
  total_samples = len(image_label_paths)

  train_split_idx = int(total_samples * train_ratio)
  valid_split_idx = train_split_idx + int(total_samples * valid_ratio)

  train_set = image_label_paths[:train_split_idx]
  valid_set = image_label_paths[train_split_idx:valid_split_idx]
  test_set = image_label_paths[valid_split_idx:]

  # Adjust for potential rounding errors to ensure all samples are included
  if len(train_set) + len(valid_set) + len(test_set) != total_samples:
    # If there's a discrepancy, add remaining samples to the test set
    remaining = total_samples - (len(train_set) + len(valid_set) + len(test_set))
    if remaining > 0:
      test_set.extend(image_label_paths[-remaining:])
    elif remaining < 0: # Should not happen with current slicing, but as a safeguard
      test_set = test_set[:remaining] # Remove excess from test_set

  return {
    'train': train_set,
    'valid': valid_set,
    'test': test_set,
  }

def create_split_directories(output_path: str):
  """
  Creates the output directories for the split dataset.

  Args:
    output_path (str): The root path for the output dataset.
  """
  if os.path.exists(output_path):
    shutil.rmtree(output_path)
  os.makedirs(output_path)

  for split_name in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(output_path, split_name, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_path, split_name, 'labels'), exist_ok=True)

def copy_split_files(
  split_data: Dict[str, List[Tuple[str, str]]],
  output_path: str,
):
  """
  Copies the split image and label files to their respective output directories.

  Args:
    split_data (Dict[str, List[Tuple[str, str]]]): Dictionary containing
                                                  'train', 'valid', 'test' splits.
    output_path (str): The root path for the output dataset.
  """
  for split_name, paths in split_data.items():
    dest_images_dir = os.path.join(output_path, split_name, 'images')
    dest_labels_dir = os.path.join(output_path, split_name, 'labels')

    for image_path, label_path in tqdm(paths, f"Copying {len(paths)} files for {split_name} split..."):
      shutil.copy(image_path, dest_images_dir)
      shutil.copy(label_path, dest_labels_dir)

def main():
  """
  Main function to orchestrate the dataset splitting process.
  """
  print(f"Starting dataset split from {DATASET_PATH} to {OUTPUT_PATH}")

  # 1. Get all image and label paths
  all_image_label_paths = get_image_label_paths(DATASET_PATH)
  print(f"Found {len(all_image_label_paths)} image-label pairs.")

  # 2. Split the dataset
  split_data = split_dataset(
    all_image_label_paths,
    TRAIN_RATIO,
    VALID_RATIO,
    TEST_RATIO,
  )

  # 3. Create output directories
  create_split_directories(OUTPUT_PATH)

  # 4. Copy files to new directories
  copy_split_files(split_data, OUTPUT_PATH)

  print("Dataset splitting complete.")

if __name__ == '__main__':
  main()
