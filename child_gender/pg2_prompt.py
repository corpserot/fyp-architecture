# BEGIN IPYTHON
from IPython.display import display
# END IPYTHON

# BEGIN MARKDOWN
# # PaliGemma2 Segmentation Annotation Batch Update
# Set up PaliGemma2 for prompting. It's used to perform batch updates to a
# YOLOv12 dataset. To achieve this, we need to convert the YOLO format into a
# readable format for PaliGemma2 and reverse it when storing them.
#
# ## Input Format Documentation
# PaliGemma2-3b-mix-448 expects 448x448 input images and text prompts.
# - For bounding box tasks, the prompt often describes the objects to detect or
#   asks for bounding box coordinates in a human-readable format.
# - For segmentation tasks, the prompt typically asks the model to produce a
#   pixel level mask or describe the regions belonging to specific object
#   classes in a clear and structured format.
#
# YOLOv12 bounding box training uses annotations in the format `[class_id,
# x_center, y_center, width, height]`, with all values normalized to the image
# dimensions. Training images are commonly resized to a fixed square resolution
# such as 640x640.
#
# For segmentation tasks, YOLOv12 typically relies on polygon based annotations
# or binary masks that correspond to each object instance, and these masks are
# trained alongside the images in the same normalized coordinate space.
# END MARKDOWN

import os # convenience

os.system('nvidia-smi') # BANG: !nvidia-smi

# Setup paths
os.chdir('/workspace') # MAGIC: %cd /workspace
home = os.getcwd()
dataset_path = "/workspace/dataset"

os.system('pip install -q gdown') # MAGIC: %pip install -q gdown
os.system('pip install -q transformers accelerate bitsandbytes pillow tqdm PyYAML hf_transfer') # MAGIC: %pip install -q transformers accelerate bitsandbytes pillow tqdm PyYAML hf_transfer
os.system('pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121') # MAGIC: %pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# NEXT CELL

os.system(f"gdown --fuzzy 'https://drive.google.com/file/d/1tvFnEur7PQjgsidcUIeYpzMSoAs--GsS/view?usp=sharing' -O {home}/dataset.zip") # BANG: !gdown --fuzzy 'https://drive.google.com/file/d/1tvFnEur7PQjgsidcUIeYpzMSoAs--GsS/view?usp=sharing' -O {home}/dataset.zip
os.system(f"mkdir -p {home}/dataset") # BANG: !mkdir -p {home}/dataset
os.system(f"rm -rf {home}/dataset") # BANG: !rm -rf {home}/dataset
os.system(f"unzip -q {home}/dataset.zip -d {home}/dataset") # BANG: !unzip -q {home}/dataset.zip -d {home}/dataset

# NEXT CELL

HF_API_KEY = "12345" # Replace with your actual Hugging Face API key
os.environ["HF_TOKEN"] = HF_API_KEY

# BEGIN MARKDOWN
# ## 1. Setup PaliGemma2
# END MARKDOWN

from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import pg2_lib # read pg2_lib.py for more information

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "google/paligemma2-3b-mix-448"
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(
  model_id, dtype=torch.bfloat16
  ).to(device) # type: ignore

# BEGIN MARKDOWN
# ## 2. YOLO to Readable Bounding Box Conversion
# END MARKDOWN

class_labels = pg2_lib.get_class_names_from_yaml(dataset_path)

# BEGIN MARKDOWN
# ## Task A: Prompting PaliGemma2 for Segmentation Polygons
# This section iterates through the dataset, prompts PaliGemma2 for polygon segmentation annotations, and updates the YOLO annotations.
# END MARKDOWN

def process_image_for_segmentation(
  image_info: pg2_lib.YoloImageInfo,
  skip_bad_annotation: bool = False,
  print_output: bool = False,
) -> list[list[float]]:
  """
  Processes a single image for segmentation using PaliGemma2 and saves the annotations.

  Args:
    image_info (YoloImageInfo): Dataclass containing image path, width, height, and YOLO annotations.
    skip_bad_annotation (bool): If True, the function returns an empty list immediately
                                if any annotation fails to parse. Defaults to False.

  Returns:
    list[list[float]]: A list of YOLO segmentation format annotations.
  """
  image_path = image_info.image_path
  image_name = os.path.basename(image_path)
  image_w, image_h = image_info.image_width, image_info.image_height
  image = Image.open(image_path).convert("RGB")
  yolo_annotations = image_info.yolo_annotations

  readable_bboxes = pg2_lib.yolo_to_readable_bbox(
  yolo_annotations,
  image_w,
  image_h,
  class_labels,
  model_input_size=448 # Pass the model's expected input size
  )
  # No longer using readable_bboxes in the prompt
  segmentation_prompt = pg2_lib.create_segmentation_prompt(class_labels)

  # Add <image> token to the prompt
  inputs_for_annotations = processor(text="<image>" + segmentation_prompt, images=image, return_tensors="pt").to(device)

  with torch.no_grad():
    output_annotations = model.generate(**inputs_for_annotations, max_new_tokens=500)

  generated_annotations_text = processor.decode(output_annotations[0], skip_special_tokens=True)
  if print_output:
    print(f"PaliGemma2 Output for {image_name}:\n{generated_annotations_text}\n")
  try:
    updated_segmentations = pg2_lib.parse_response_to_yolo_segmentation(
      generated_annotations_text,
      image_w,
      image_h,
      class_labels,
      model_input_size=448, # Pass the model's expected input size
      fail_on_error=skip_bad_annotation
    )
  except Exception as e:
    print(f"Error parsing PaliGemma2 output: {e}. Skipping annotation update for this image.")
    updated_segmentations = []

  output_label_path = os.path.join(dataset_path, "labels", os.path.splitext(image_name)[0] + ".txt")
  if updated_segmentations:
    with open(output_label_path, 'w') as f:
      for ann in updated_segmentations:
        f.write(f"{int(ann[0])} {' '.join(map(str, ann[1:]))}\n")
  return updated_segmentations

# The save_segmentations_to_file function is moved to pg2_lib.py
# def save_segmentations_to_file(...) -> None:
#   ...

def display_segmentation(
  image_info: pg2_lib.YoloImageInfo,
  segmentations: list[list[float]],
  display_width: int,
) -> None:
  """
  Displays an image with drawn segmentation polygons.

  Args:
    image_info (YoloImageInfo): Dataclass containing image path, width, height.
    segmentations (list of lists): A list of YOLO segmentation format annotations.
    display_width (int): Width to display the image.
  """
  image = Image.open(image_info.image_path).convert("RGB")
  image = pg2_lib.draw_segmentations_on_image(image, segmentations, class_labels)
  if display_width:
    # Calculate height to maintain aspect ratio
    aspect_ratio = image.height / image.width
    display_height = int(display_width * aspect_ratio)
    image = image.resize((display_width, display_height), Image.Resampling.LANCZOS)
  display(image)

# NEXT CELL

dataset_info = pg2_lib.read_yolo_dataset(dataset_path)
assert dataset_info, "No dataset information?!"

try_images = (0,10)
num_try_images = try_images[1] - try_images[0]
sample_images_with_segmentations: list[tuple[pg2_lib.YoloImageInfo, list[list[float]]]] = []

for i, image_info in tqdm(enumerate(dataset_info[try_images[0]:try_images[1]])):
  segmentations = process_image_for_segmentation(image_info, skip_bad_annotation=False, print_output=True)
  sample_images_with_segmentations.append((image_info, segmentations))

for image_info, segmentations in sample_images_with_segmentations:
  if segmentations:
    display_segmentation(image_info, segmentations, display_width=600)

# NEXT CELL

dataset_info = pg2_lib.read_yolo_dataset(dataset_path)
assert dataset_info, "No dataset information?!"

for i, image_info in tqdm(enumerate(dataset_info)):
  segmentations = process_image_for_segmentation(image_info, skip_bad_annotation=False, print_output=False)
  image_name = os.path.basename(image_info.image_path)
  pg2_lib.save_segmentations_to_file(image_name, dataset_path, segmentations)
