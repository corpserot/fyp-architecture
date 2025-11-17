import os
import numpy as np

SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = f'{SCRIPTDIR}/dataset_raw'

# Constants for split names and file extensions
SPLIT_NAMES = ['train', 'valid', 'test']
LABEL_FILE_EXT = '.txt'

def validate_bbox_format(coordinates: list[float], label_path: str, line_num: int, line_content: str):
  """Validates if coordinates conform to YOLO bounding box format."""
  if len(coordinates) != 4: # class_id is already parsed, so expect 4 coords
    print(f"Error in file: {label_path}, Line {line_num}: Bounding box format expects 4 coordinates, got {len(coordinates)}. Line content: '{line_content}'")
    return False
  x_center, y_center, width, height = coordinates
  if not (0.0 <= x_center <= 1.0 and
      0.0 <= y_center <= 1.0 and
      0.0 <= width <= 1.0 and
      0.0 <= height <= 1.0):
    print(f"Error in file: {label_path}, Line {line_num}: Bounding box coordinates out of [0.0, 1.0] range. Line content: '{line_content}'")
    return False
  return True

def validate_segmentation_format(coordinates: list[float], label_path: str, line_num: int, line_content: str):
  """Validates if coordinates conform to YOLO segmentation format."""
  if len(coordinates) < 4 or len(coordinates) % 2 != 0:
    print(f"Error in file: {label_path}, Line {line_num}: Segmentation format expects an even number of coordinates (>=4). Got {len(coordinates)}. Line content: '{line_content}'")
    return False
  if not all(0.0 <= coord <= 1.0 for coord in coordinates):
    print(f"Error in file: {label_path}, Line {line_num}: Segmentation polygon coordinates out of [0.0, 1.0] range. Line content: '{line_content}'")
    return False
  return True

def validate_annotation(label_path: str):
  annotation_type: str | None = None
  with open(label_path, 'r') as f:
    for line_num, raw_line in enumerate(f.readlines()):
      line = raw_line.strip()
      if not line: # Skip empty lines
        continue

      parts = line.split()
      if not parts: # Skip empty lines after stripping
        continue

      try:
        # class_id = int(parts[0]) # Class ID is always the first part. Unused for now
        coordinates = list(map(float, parts[1:])) # Remaining parts are coordinates

        # Determine format based on number of coordinates
        coords_count = len(coordinates)
        if coords_count == 4:
          if annotation_type is None:
            annotation_type = "bounding box"
          elif annotation_type != "bounding box":
            print(f"Error in file: {label_path}: Mixing bounding box and segmentation types")
            continue
          validate_bbox_format(coordinates, label_path, line_num + 1, line)
        elif coords_count >= 4 and coords_count % 2 == 0:
          if annotation_type is None:
            annotation_type = "segmentation"
          elif annotation_type != "segmentation":
            print(f"Error in file: {label_path}: Mixing bounding box and segmentation types")
            continue
          validate_segmentation_format(coordinates, label_path, line_num + 1, line)
        else:
          print(f"Error in file: {label_path}, Line {line_num + 1}: Incorrect number of values for either bbox (5 total) or segmentation (class_id + even number of points >= 4). Line content: '{line}'")
          continue

      except ValueError:
        print(f"Error in file: {label_path}, Line {line_num + 1}: Non-numeric value found. Line content: '{line}'")
      except Exception as e:
        print(f"Unexpected error in file: {label_path}, Line {line_num + 1}: {e}. Line content: '{line}'")

def find_yolo_annotation_errors(dataset_path: str):
  """
  Finds and reports errors in YOLO format annotation files, supporting both
  bounding box (class_id + 4 values) and segmentation (class_id + polygon points) formats.
  """
  print(f"Searching for annotation errors in: {dataset_path}")

  for split_name in SPLIT_NAMES:
    labels_path = os.path.join(dataset_path, split_name, 'labels')
    if not os.path.exists(labels_path):
      print(f"Warning: Label directory not found for {split_name}: {labels_path}")
      continue

    print(f"\nChecking {split_name} split labels...")
    label_files = [f for f in os.listdir(labels_path) if f.lower().endswith(LABEL_FILE_EXT)]

    for label_file in label_files:
      validate_annotation(os.path.join(labels_path, label_file))

  print("\nAnnotation error checking complete.")

if __name__ == '__main__':
  find_yolo_annotation_errors(DATASET_PATH)
