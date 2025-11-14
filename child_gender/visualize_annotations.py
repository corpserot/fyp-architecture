import os
from typing import Sequence
import cv2
import numpy as np
import colorsys
import yaml
from tqdm import tqdm

SCRIPTDIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(SCRIPTDIR, 'dataset_processed')
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

type RGB = tuple[int, int, int];
type BBOX = tuple[int, int, int, int, int];

def get_distinct_colors(n: int) -> list[RGB]:
    """Generate n distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
        # Using a fixed saturation and lightness for better visibility
        lightness = (50 + (i % 2) * 10) / 100.0
        saturation = (90 - (i % 3) * 10) / 100.0
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(tuple(int(x * 255) for x in rgb))
    return colors
_distinct_colors: list[RGB] = []
_distinct_colors_count = 0
def set_distinct_colors(n: int):
    global _distinct_colors, _distinct_colors_count
    _distinct_colors_count = n
    _distinct_colors = get_distinct_colors(n)

def get_color_for_class(class_id: int):
    """Returns a consistent color for a given class_id."""
    return _distinct_colors[class_id % _distinct_colors_count]

def parse_yolo_annotation(annotation_path: str, img_width: int, img_height: int) -> list[BBOX]:
    """
    Parses a YOLO annotation file and returns bounding box coordinates.
    Annotation format: class_id center_x center_y width height (normalized)
    Returns a list of (class_id, x_min, y_min, x_max, y_max) in pixel coordinates.
    """
    boxes = []
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            center_x, center_y, width, height = map(float, parts[1:])

            # Convert normalized coordinates to pixel coordinates
            x_center = int(center_x * img_width)
            y_center = int(center_y * img_height)
            w = int(width * img_width)
            h = int(height * img_height)

            x_min = int(x_center - w / 2)
            y_min = int(y_center - h / 2)
            x_max = int(x_center + w / 2)
            y_max = int(y_center + h / 2)

            boxes.append((class_id, x_min, y_min, x_max, y_max))
    return boxes

def draw_bounding_boxes(image_path: str, annotation_path: str, output_path: str, class_names: Sequence[str] | None):
    """
    Reads an image, draws bounding boxes from YOLO annotations, and saves the result.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    img_height, img_width, _ = img.shape
    boxes = parse_yolo_annotation(annotation_path, img_width, img_height)

    for class_id, x_min, y_min, x_max, y_max in boxes:
        color = get_color_for_class(class_id)
        thickness = 2
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

        if class_names and class_id < len(class_names):
            label = class_names[class_id]
        else:
            label = f"Class {class_id}"

        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        # Determine text position (inside the box, top-left with some padding)
        text_x = x_min + 5
        text_y = y_min + text_height + 5

        # Ensure text is within image bounds
        if text_y > img_height:
            text_y = y_max - 5
        if text_x + text_width > img_width:
            text_x = x_max - text_width - 5

        # Draw semi-transparent background for text
        # Background color (black with 50% opacity)
        bg_color = (0, 0, 0) # Black
        alpha = 0.5

        overlay = img.copy()
        cv2.rectangle(overlay, (text_x - 2, text_y - text_height - 2), (text_x + text_width + 2, text_y + baseline + 2), bg_color, -1)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Draw text
        cv2.putText(img, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA) # White text

    cv2.imwrite(output_path, img)

    # it's gonna be too noisy
    # print(f"Visualized {os.path.basename(image_path)} and saved to {output_path}")

def visualize_dataset(split_type: str, dataset_dir: str, class_names: Sequence[str] | None = None):
    """
    Visualizes YOLO annotations for all images in a dataset directory.
    """
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    output_dir = os.path.join(dataset_dir, 'visualized')

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found at {images_dir}")
        return
    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory not found at {labels_dir}")
        return

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(IMAGE_EXTS)]
    image_files.sort()

    for image_file in tqdm(image_files, desc=f"Visualizing {split_type} annotated images"):
        image_path = os.path.join(images_dir, image_file)

        # label file has the same name as image file but with .txt extension
        label_file_name = os.path.splitext(image_file)[0] + '.txt'
        annotation_path = os.path.join(labels_dir, label_file_name)

        if os.path.exists(annotation_path):
            output_path = os.path.join(output_dir, image_file)
            draw_bounding_boxes(image_path, annotation_path, output_path, class_names)
        else:
            print(f"Warning: Annotation file not found for {image_file} at {annotation_path}")

if __name__ == "__main__":
    loaded_class_names: list[str] | None = None
    class_names_file = os.path.join(DATASET_PATH, "data.yaml")
    try:
        with open(class_names_file, 'r') as f:
            data = yaml.safe_load(f)
            if 'names' in data and isinstance(data['names'], list):
                loaded_class_names = data['names']
                print(f"Loaded class names from {class_names_file}: {loaded_class_names}")
            else:
                raise Exception(f"Error: 'names' key not found or not a list in {class_names_file}")
    except FileNotFoundError:
        raise Exception(f"Error: Class names file not found at {class_names_file}")
    except yaml.YAMLError as e:
        raise Exception(f"Error: Error parsing YAML from {class_names_file}: {e}")
    except Exception as e:
        raise Exception(f"Error: An unexpected error occurred loading class names from {class_names_file}: {e}")

    if loaded_class_names is None or not loaded_class_names:
        raise Exception("Error: No class names found")

    set_distinct_colors(len(loaded_class_names));

    for split in ('train', 'test', 'valid'):
        visualize_dataset(split, os.path.join(DATASET_PATH, split), loaded_class_names)
