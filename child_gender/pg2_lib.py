
import yaml
import os
import re
from PIL import Image, ImageDraw
from dataclasses import dataclass

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

@dataclass
class YoloImageInfo:
  image_path: str
  image_width: int
  image_height: int
  yolo_annotations: list[list[float]]

def get_class_names_from_yaml(dataset_path: str):
  """Reads class names from the data.yaml file in the dataset_path."""
  yaml_path = os.path.join(dataset_path, 'data.yaml')
  with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)
  return data.get('names', [])

def read_yolo_dataset(dataset_path: str):
  """
  Reads images and YOLO annotations from a specified dataset directory.

  Args:
    dataset_path (str): The path to the dataset directory, which should contain
              'images/' and 'labels/' subdirectories.

  Returns:
    list[YoloImageInfo]: A list of YoloImageInfo dataclass instances.
  """
  images_dir = os.path.join(dataset_path, 'images')
  labels_dir = os.path.join(dataset_path, 'labels')

  dataset_info: list[YoloImageInfo] = []
  image_filenames = os.listdir(images_dir)
  image_filenames.sort()
  for image_filename in image_filenames:
    if image_filename.lower().endswith(IMAGE_EXTS):
      image_path = os.path.join(images_dir, image_filename)
      label_path = os.path.join(labels_dir, os.path.splitext(image_filename)[0] + '.txt')

      if os.path.exists(label_path):
        with Image.open(image_path) as img:
          image_width, image_height = img.size

        yolo_annotations: list[list[float]] = []
        with open(label_path, 'r') as f:
          for line in f:
            parts = list(map(float, line.strip().split()))
            # Skip segmentation polygons (more than 5 parts) and malformed annotations
            if len(parts) == 5:
              yolo_annotations.append(parts)

        dataset_info.append(
          YoloImageInfo(
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            yolo_annotations=yolo_annotations
          )
        )
      else:
        print(f"Warning: No label file found for {image_filename} at {label_path}")
  return dataset_info

def yolo_to_readable_bbox(
  yolo_annotations: list[list[float]],
  image_width: int,
  image_height: int,
  class_names: list[str],
  model_input_size: int = 448, # Default to 448 as per PaliGemma2-3b-mix-448
) -> str:
  """
  Converts YOLO format annotations to a human-readable string for PaliGemma2.

  Args:
    yolo_annotations (list of lists): Each inner list is [class_id, x_center, y_center, width, height]
                      where coordinates are normalized (0 to 1).
    image_width (int): The width of the original image.
    image_height (int): The height of the original image.
    class_names (list of str): A list of class names, where index corresponds to class_id.

  Returns:
    str: A human-readable string describing the bounding boxes.
  """
  readable_boxes = []
  for annotation in yolo_annotations:
    class_id, x_center_norm, y_center_norm, width_norm, height_norm = annotation

    # Denormalize coordinates
    x_center = x_center_norm * image_width
    y_center = y_center_norm * image_height
    width = width_norm * image_width
    height = height_norm * image_height

    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)

    # Scale coordinates to model_input_size
    scale_x = model_input_size / image_width
    scale_y = model_input_size / image_height

    x_min_scaled = int(x_min * scale_x)
    y_min_scaled = int(y_min * scale_y)
    x_max_scaled = int(x_max * scale_x)
    y_max_scaled = int(y_max * scale_y)

    class_name = class_names[int(class_id)]
    readable_boxes.append(f"class object '{class_name}' box at [x_min={x_min_scaled}, y_min={y_min_scaled}, x_max={x_max_scaled}, y_max={y_max_scaled}]")

  return "; ".join(readable_boxes)


def parse_response_to_yolo_bbox(
  paligemma_output: str,
  image_width: int,
  image_height: int,
  class_names: list[str],
  model_input_size: int = 448, # Default to 448 as per PaliGemma2-3b-mix-448
) -> list[list[float]]:
  """
  Parses PaliGemma2's free-form text output into YOLO format annotations.

  Args:
    paligemma_output (str): The text output from PaliGemma2 describing bounding boxes.
    image_width (int): The width of the original image.
    image_height (int): The height of the original image.
    class_names (list of str): A list of class names, where index corresponds to class_id.
    model_input_size (int): The input size of the model (default 448).

  Returns:
    list of lists: A list of YOLO format annotations: [class_id, x_center, y_center, width, height].
  """
  yolo_annotations = []
  # Example expected format: "class object 'person' at [x_min=100, y_min=100, x_max=200, y_max=200]; ..."
  pattern = r"class object '([^']+)' at \[x_min=(\d+), y_min=(\d+), x_max=(\d+), y_max=(\d+)\]"
  matches = re.findall(pattern, paligemma_output)

  for match in matches:
    class_name_str, x_min_str, y_min_str, x_max_str, y_max_str = match
    try:
      class_id = class_names.index(class_name_str)
    except ValueError:
      print(f"Warning: Class name '{class_name_str}' not found in provided class_names. Skipping this annotation.")
      continue

    x_min_model, y_min_model, x_max_model, y_max_model = int(x_min_str), int(y_min_str), int(x_max_str), int(y_max_str)

    # Scale from model input size to original image dimensions
    scale_x = image_width / model_input_size
    scale_y = image_height / model_input_size

    x_min = x_min_model * scale_x
    y_min = y_min_model * scale_y
    x_max = x_max_model * scale_x
    y_max = y_max_model * scale_y

    # Convert to YOLO format
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + width / 2
    y_center = y_min + height / 2

    x_center_norm = x_center / image_width
    y_center_norm = y_center / image_height
    width_norm = width / image_width
    height_norm = height / image_height

    yolo_annotations.append([
      float(class_id),
      round(x_center_norm, 6),
      round(y_center_norm, 6),
      round(width_norm, 6),
      round(height_norm, 6),
    ])
  return yolo_annotations

def parse_response_to_yolo_segmentation(
  paligemma_output: str,
  image_width: int,
  image_height: int,
  class_names: list[str],
  model_input_size: int = 448, # Default to 448 as per PaliGemma2-3b-mix-448
  fail_on_error: bool = False,
) -> list[list[float]]:
  """
  Parses PaliGemma2's free-form text output into YOLO segmentation format annotations.

  Args:
    paligemma_output (str): The text output from PaliGemma2 describing polygons.
    image_width (int): The width of the original image.
    image_height (int): The height of the original image.
    class_names (list of str): A list of class names, where index corresponds to class_id.
    fail_on_error (bool): If True, the function returns an empty list immediately
                          if any annotation fails to parse. Defaults to False.

  Returns:
    list of lists: A list of YOLO segmentation format annotations:
                   [class_id, x1_norm, y1_norm, x2_norm, y2_norm, ...].
  """
  yolo_segmentations = []

  # The new prompt explicitly defines the output format:
  # "object: 'class_name' polygon: <segXXX><segXXX>...; "
  # We need to split by ';' and then parse each object.

  # Split the output into individual object descriptions
  object_descriptions = paligemma_output.split(';')

  # Regex to extract class name and seg tokens for each object
  # Pattern: "object: 'class_name' polygon: <segXXX><segXXX>..."
  object_pattern = r"object:\s*'([^']+)'\s*polygon:\s*((?:<seg\d+>\s*)+)"
  seg_token_pattern = r"<seg(\d+)>" # For extracting individual seg values

  for desc in object_descriptions:
    if not desc.strip():
      continue # Skip empty descriptions

    match = re.search(object_pattern, desc)
    if not match:
      print(f"Warning: Could not parse object description: '{desc}'. Skipping.")
      if fail_on_error:
        return []
      continue

    class_name = match.group(1)
    seg_tokens_string = match.group(2)

    try:
      class_id = class_names.index(class_name)
    except ValueError:
      print(f"Warning: Class name '{class_name}' not found in provided class_names. Skipping this object.")
      if fail_on_error:
        return []
      continue

    # Extract individual seg values from the seg_tokens_string
    seg_values_str = re.findall(seg_token_pattern, seg_tokens_string)

    if not seg_values_str:
      print(f"Warning: No <segXXX> tokens found for object '{class_name}'. Skipping.")
      if fail_on_error:
        return []
      continue

    try:
      coords_model = [float(val) for val in seg_values_str]
    except ValueError as e:
      print(f"Warning: Could not parse <segXXXX> token values for '{class_name}': {e}. Skipping.")
      if fail_on_error:
        return []
      continue

    if len(coords_model) % 2 != 0:
      print(f"Warning: Malformed polygon coordinates (odd number of values) for '{class_name}'. Skipping.")
      if fail_on_error:
        return []
      continue

    normalized_coords: list[float] = [float(class_id)]
    for i in range(0, len(coords_model), 2):
      x_model = coords_model[i]
      y_model = coords_model[i + 1]

      # Scale coordinates from model input size to original image dimensions
      x_scaled = x_model * (image_width / model_input_size)
      y_scaled = y_model * (image_height / model_input_size)

      x_norm = round(x_scaled / image_width, 6)
      y_norm = round(y_scaled / image_height, 6)
      normalized_coords.extend([x_norm, y_norm])

    if len(normalized_coords) > 1: # Ensure there are actual polygon points
      yolo_segmentations.append(normalized_coords)

  return yolo_segmentations

def create_segmentation_prompt(
  class_labels: list[str],
) -> str:
  """
  Creates a segmentation prompt for PaliGemma2 to segment and identify objects.

  Args:
    class_labels (list[str]): A list of class names to include in the prompt.

  Returns:
    str: The complete segmentation prompt.
  """
  # Aligning with user's feedback for short, brief, and simple instructions.
  # The model should segment all objects and identify them, referencing the provided class names.
  # Explicitly define the output format for multiple objects.
  return (
    f"segment these objects: {', '.join(class_labels)}. "
    "Output format: object: 'class_name' polygon: <segXXX><segXXX>...; "
    "Repeat for each object."
  )

def save_segmentations_to_file(
  image_name: str,
  dataset_path: str,
  updated_segmentations: list[list[float]],
) -> None:
  """
  Saves YOLO segmentation annotations to a label file.

  Args:
    image_name (str): The base name of the image file (e.g., "image.jpg").
    dataset_path (str): The path to the dataset directory.
    updated_segmentations (list of lists): A list of YOLO segmentation format annotations.
  """
  output_label_path = os.path.join(dataset_path, "labels", os.path.splitext(image_name)[0] + ".txt")
  if updated_segmentations:
    with open(output_label_path, 'w') as f:
      for ann in updated_segmentations:
        f.write(f"{int(ann[0])} {' '.join(map(str, ann[1:]))}\n")
  else:
    print(f"No valid segmentations parsed for {image_name}. Skipping file write.")

# A list of distinct colors in RGB format
COLORS = [
  (255, 99, 71),   # Tomato
  (60, 179, 113),  # MediumSeaGreen
  (65, 105, 225),  # RoyalBlue
  (255, 165, 0),   # Orange
  (147, 112, 219), # MediumPurple
  (0, 191, 255),   # DeepSkyBlue
  (255, 20, 147),  # DeepPink
  (0, 255, 0),     # Lime
  (255, 255, 0),   # Yellow
  (0, 255, 255),   # Cyan
]
NUM_COLORS = len(COLORS)
def _get_color_for_class(class_id: int) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
  """
  Generates a consistent color for a given class_id.

  Args:
    class_id (int): The ID of the class.

  Returns:
    tuple[tuple[int, int, int], tuple[int, int, int, int]]: A tuple containing
    (outline_color_rgb, fill_color_rgba).
  """
  outline_rgb = COLORS[class_id % NUM_COLORS]
  # Use a semi-transparent version for the fill (e.g., 20% opacity)
  fill_rgba = outline_rgb + (51,) # 51 is approx 20% of 255

  return outline_rgb, fill_rgba

def draw_segmentations_on_image(
  image: Image.Image,
  yolo_segmentations: list[list[float]],
  class_names: list[str],
) -> Image.Image:
  """
  Draws YOLO segmentation annotations (polygons) on an image.

  Args:
    image (PIL.Image.Image): The input image.
    yolo_segmentations (list of lists): A list of YOLO segmentation format annotations:
                                       [class_id, x1_norm, y1_norm, x2_norm, y2_norm, ...].
    class_names (list of str): A list of class names, where index corresponds to class_id.

  Returns:
    PIL.Image.Image: The image with drawn segmentation polygons.
  """
  draw = ImageDraw.Draw(image)
  image_width, image_height = image.size

  for segmentation in yolo_segmentations:
    class_id = int(segmentation[0])
    normalized_coords = segmentation[1:]

    if class_id < 0 or class_id >= len(class_names):
      print(f"Warning: Invalid class_id {class_id}. Skipping segmentation.")
      continue

    # Denormalize coordinates
    polygon_points: list[tuple[float, float]] = []
    for i in range(0, len(normalized_coords), 2):
      x = normalized_coords[i] * image_width
      y = normalized_coords[i+1] * image_height
      polygon_points.append((x, y))

    if len(polygon_points) > 1:
      outline_color, fill_color = _get_color_for_class(class_id)

      # Draw polygon
      draw.polygon(polygon_points, outline=outline_color, width=2)
      # Draw a semi-transparent fill
      draw.polygon(polygon_points, fill=fill_color)

      # Draw class name with a semi-transparent white background
      first_point_x, first_point_y = polygon_points[0]
      class_name = class_names[class_id]

      # Calculate text size and position for background
      text_bbox = draw.textbbox((first_point_x, first_point_y - 10), class_name)
      text_x_min, text_y_min, text_x_max, text_y_max = text_bbox

      # Add some padding to the background
      padding = 2
      bg_x_min = text_x_min - padding
      bg_y_min = text_y_min - padding
      bg_x_max = text_x_max + padding
      bg_y_max = text_y_max + padding

      # Draw semi-transparent white background rectangle
      draw.rectangle((bg_x_min, bg_y_min, bg_x_max, bg_y_max), fill=(255, 255, 255, 51))

      # Draw class name
      draw.text((first_point_x, first_point_y - 10), class_name, fill=outline_color)

  return image
