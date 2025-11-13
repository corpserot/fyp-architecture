import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------- configs --------------------------------- #

IMG_SIZE = 640 # Target image size for preprocessing
SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = f'{SCRIPTDIR}/dataset_raw'
OUTPUT_PATH = f'{SCRIPTDIR}/dataset_processed'

IMG_EXT = ('.jpg', '.jpeg', '.png')

# this pipeline need to be used during inference
def preprocess_pipeline(img_size=IMG_SIZE):
    return A.Compose([
        # pyright: ignore[reportArgumentType]
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def augmentation_pipeline(img_size=IMG_SIZE):
    """Defines the augmentation pipeline using Albumentations."""
    return A.Compose([
        # pyright: ignore[reportArgumentType]
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(p=0.2),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-15, 15), p=0.5),
        # use size=(width, height). it is correct
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.2),
        # A.Normalize(), # Removed as it causes black images when saving with cv2.imwrite
        # ToTensorV2() # This is for PyTorch, might not be needed for saving raw images
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# ---------------------------------- script ---------------------------------- #

# Create output directories if they don't exist
os.makedirs(os.path.join(OUTPUT_PATH, 'train', 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'train', 'labels'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'valid', 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'valid', 'labels'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'test', 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'test', 'labels'), exist_ok=True)

print(f"Input dataset path: {DATASET_PATH}")
print(f"Output processed dataset path: {OUTPUT_PATH}")

def read_yolo_annotations(label_path, img_width, img_height):
    """Reads YOLO format annotations from a file."""
    boxes = []
    assert os.path.exists(label_path), f"{label_path} does not exist"
    with open(label_path, 'r') as f:
        for line_num, line in enumerate(f.readlines()): # line_num is now correctly defined here
            line = line.strip()
            if not line: # Skip empty lines
                continue

            parts_float = list(map(float, line.split()))
            class_id = int(parts_float[0])
            coordinates_normalized = parts_float[1:]

            x_min_raw, y_min_raw, x_max_raw, y_max_raw = 0, 0, 0, 0 # Initialize

            # Determine if bounding box or segmentation format
            if len(coordinates_normalized) == 4: # YOLO bounding box format: x_center, y_center, width, height
                x_center, y_center, width, height = coordinates_normalized
                x_min_raw = (x_center - width / 2) * img_width
                y_min_raw = (y_center - height / 2) * img_height
                x_max_raw = (x_center + width / 2) * img_width
                y_max_raw = (y_center + height / 2) * img_height
            elif len(coordinates_normalized) >= 4 and len(coordinates_normalized) % 2 == 0: # YOLO segmentation format: x1 y1 x2 y2 ...
                x_coords = [coordinates_normalized[i] * img_width for i in range(0, len(coordinates_normalized), 2)]
                y_coords = [coordinates_normalized[i+1] * img_height for i in range(0, len(coordinates_normalized), 2)]
                x_min_raw = min(x_coords)
                y_min_raw = min(y_coords)
                x_max_raw = max(x_coords)
                y_max_raw = max(y_coords)
            else:
                print(f"Warning: Unknown annotation format in {label_path}, Line {line_num + 1}. Skipping. Line content: '{line}'")
                continue # Skip this line if format is unknown

            # Add assertion for significantly out-of-range values before clamping
            # This helps catch truly malformed bbox data, not just minor floating point errors
            check_x = lambda x : -(0.02 * img_width) <= x <= (1.02 * img_width)
            check_y = lambda y : -(0.02 * img_height) <= y <= (1.02 * img_height)
            assert check_x(x_min_raw), f"x_min_raw ({x_min_raw}) significantly out of range for '{label_path}' bbox: {line.strip()}"
            assert check_y(y_min_raw), f"y_min_raw ({y_min_raw}) significantly out of range for '{label_path}' bbox: {line.strip()}"
            assert check_x(x_max_raw), f"x_max_raw ({x_max_raw}) significantly out of range for '{label_path}' bbox: {line.strip()}"
            assert check_y(y_max_raw), f"y_max_raw ({y_max_raw}) significantly out of range for '{label_path}' bbox: {line.strip()}"

            # Clamp values to be within image boundaries [0, img_width/height]
            x_min = max(0, x_min_raw)
            y_min = max(0, y_min_raw)
            x_max = min(img_width, x_max_raw)
            y_max = min(img_height, y_max_raw)

            assert (x_max - x_min) > 0.01 * img_width, f"width ({x_max - x_min}) is too small for '{label_path}' bbox: {line.strip()}"
            assert (y_max - y_min) > 0.01 * img_height, f"height ({y_max - y_min}) is too small for '{label_path}' bbox: {line.strip()}"

            boxes.append([x_min, y_min, x_max, y_max, class_id])
    assert len(boxes) > 0, f"annotation is empty for '{label_path}'"
    return np.array(boxes)

def write_yolo_annotations(output_label_path, boxes, img_width, img_height):
    """Writes YOLO format annotations to a file."""
    with open(output_label_path, 'w') as f:
        for box in boxes:
            x_min, y_min, x_max, y_max, class_id = box

            # Convert (x_min, y_min, x_max, y_max) to normalized (x_center, y_center, width, height)
            x_center = ((x_min + x_max) / 2) / img_width
            y_center = ((y_min + y_max) / 2) / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def load_and_preprocess_split(pp_pipeline, split_name):
    """Loads and preprocesses images and labels for a given split."""
    print(f"\nProcessing {split_name} split...")
    images_path = os.path.join(DATASET_PATH, split_name, 'images')
    labels_path = os.path.join(DATASET_PATH, split_name, 'labels')
    image_files = [f for f in os.listdir(images_path) if f.lower().endswith(IMG_EXT)]

    processed_data = []

    for img_file in tqdm(image_files, desc=f"Loading {split_name} images"):
        label_path = os.path.join(labels_path, os.path.splitext(img_file)[0] + '.txt')
        image = cv2.imread(os.path.join(images_path, img_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        boxes = read_yolo_annotations(label_path, width, height)

        transformed = pp_pipeline(image=image, bboxes=boxes[:, :4], class_labels=boxes[:, 4])
        resized_image = transformed['image']
        resized_boxes = np.array(transformed['bboxes'])
        class_labels = np.array(transformed['class_labels'])

        # Combine resized boxes with class labels
        if len(resized_boxes) > 0:
            resized_boxes_with_classes = np.hstack((resized_boxes, class_labels.reshape(-1, 1)))
        else:
            resized_boxes_with_classes = np.array([])

        processed_data.append({
            'image': resized_image,
            'boxes': resized_boxes_with_classes,
            'file_name': img_file
        })
    return processed_data

def save_single_image_and_labels(image, boxes, file_name, output_base_path, split_name):
    """Saves a single image and its labels to the specified output directory."""
    output_images_path = os.path.join(output_base_path, split_name, 'images')
    output_labels_path = os.path.join(output_base_path, split_name, 'labels')

    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_labels_path, exist_ok=True)

    img_output_path = os.path.join(output_images_path, file_name)
    label_output_path = os.path.join(output_labels_path, os.path.splitext(file_name)[0] + '.txt')

    # Save image (Albumentations returns RGB, cv2 expects BGR for saving)
    cv2.imwrite(img_output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Save labels
    if len(boxes) > 0:
        write_yolo_annotations(label_output_path, boxes, IMG_SIZE, IMG_SIZE)
    else:
        # Create an empty label file if no boxes, or if it's an augmented image with no boxes
        open(label_output_path, 'a').close() # Create empty file

def augment_and_save_data(data_list, output_base_path, split_name, aug_pipeline=None, aug_per_image=1):
    """Applies augmentation to a list of image-bbox data and saves them incrementally."""
    if aug_pipeline is None:
        aug_pipeline = augmentation_pipeline()

    for item in tqdm(data_list, desc=f"Augmenting and saving {split_name} data"):
        image = item['image']
        boxes = item['boxes']
        file_name = item['file_name']

        base_name, ext = os.path.splitext(file_name)

        # Save original image
        save_single_image_and_labels(image, boxes, f"{base_name}_orig{ext}", output_base_path, split_name)

        for i in range(aug_per_image):
            input_bboxes = boxes[:, :4] if len(boxes) > 0 else []
            input_class_labels = boxes[:, 4] if len(boxes) > 0 else []

            transformed = aug_pipeline(
                image=image,
                bboxes=input_bboxes,
                class_labels=input_class_labels
            )

            aug_image = transformed['image']
            aug_boxes = np.array(transformed['bboxes'])
            aug_class_labels = np.array(transformed['class_labels'])

            if len(aug_boxes) > 0:
                aug_boxes_with_classes = np.hstack((aug_boxes, aug_class_labels.reshape(-1, 1)))
            else:
                aug_boxes_with_classes = np.array([])

            save_single_image_and_labels(aug_image, aug_boxes_with_classes, f"{base_name}_aug{i}{ext}", output_base_path, split_name)

# Main execution block
if __name__ == '__main__':
    pp_pipeline = preprocess_pipeline()
    aug_pipeline = augmentation_pipeline()

    train_data_preprocessed = load_and_preprocess_split(pp_pipeline, 'train')
    valid_data_preprocessed = load_and_preprocess_split(pp_pipeline, 'valid')
    test_data_preprocessed = load_and_preprocess_split(pp_pipeline, 'test')

    # Augment and save train data incrementally
    augment_and_save_data(train_data_preprocessed, OUTPUT_PATH, 'train', A.Compose([]), aug_per_image=0)

    # Save valid and test data directly (no augmentation for these splits)
    # For valid and test, we just save the preprocessed data.
    # The `augment_and_save_data` function can be used with aug_per_image=0
    # to just save the original preprocessed images.
    augment_and_save_data(valid_data_preprocessed, OUTPUT_PATH, 'valid', A.Compose([]), aug_per_image=0)
    augment_and_save_data(test_data_preprocessed, OUTPUT_PATH, 'test', A.Compose([]), aug_per_image=0)

    print("\nPreprocessing and augmentation complete!")
