# BEGIN IPYTHON
from IPython.display import display
# END IPYTHON

# BEGIN MARKDOWN
# # PaliGemma2 Bounding Box Detection Annotation Batch Update
# Set up PaliGemma2 for prompting. It's used to perform batch updates to a
# YOLOv12 dataset using bounding box detection. To achieve this, we prompt for
# detections and parse them into YOLO format.
#
# ## Input Format
# PaliGemma2-3b-mix-448 expects 448x448 input images and text prompts.
# - For bounding box tasks, the prompt describes the objects to detect.
# - The special token <image> MUST be included in the prompt
#
# ## Output Format
# PaliGemma2's outputs are text-only. The way it is able to precisely express
# bounding boxes is through using special tokens <locXXXX> with XXXX being value
# within 0 and 1023. Bounding boxes will have y_min, x_min, y_max, x_max. For
# instance, an example prompt "detect all cat ; dog" may yield:
#
# ```
# <loc0123><loc0456><loc0789><loc1023> cat;
# <loc0023><loc0656><loc0389><loc0923> dog;
# <loc0050><loc0050><loc0500><loc0500> dod;
# ```
#
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

os.system(f"gdown --fuzzy 'https://drive.google.com/file/d/1CBhRn6I4bUbxy5Z3OHmupL4qIOF24Pir/view?usp=sharing' -O {home}/dataset.zip") # BANG: !gdown --fuzzy 'https://drive.google.com/file/d/1CBhRn6I4bUbxy5Z3OHmupL4qIOF24Pir/view?usp=sharing' -O {home}/dataset.zip
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

# Load YOLO model once
yolo_model_path = os.path.join(home, 'my-yolov12s.pt')
yolo_model = pg2_lib.load_yolo_model(yolo_model_path)

# BEGIN MARKDOWN
# ## Task A: Prompting PaliGemma2 for Bounding Box Detection
# This section iterates through the dataset, prompts PaliGemma2 for bounding box detections, and updates the YOLO annotations.
# END MARKDOWN

def process_image_for_detection(
  image_info: pg2_lib.YoloImageInfo,
  yolo_model: pg2_lib.YOLO,
  class_labels: list[str],
  skip_bad_annotation: bool = False,
  print_output: bool = False,
  save_to_file: bool = True,
) -> list[list[float]]:
  """
  Processes a single image for bounding box detection using PaliGemma2.

  Args:
    image_info (YoloImageInfo): Dataclass containing image path, width, height, and YOLO annotations.
    yolo_model (pg2_lib.YOLO): The loaded YOLO model for counting detections.
    class_labels (list[str]): A list of class names.
    skip_bad_annotation (bool): If True, the function returns an empty list immediately
                                if any annotation fails to parse. Defaults to False.
    print_output (bool): If True, print the generated text. Defaults to False.
    save_to_file (bool): If True, save annotations to label file. Defaults to True.

  Returns:
    list[list[float]]: A list of YOLO bounding box format annotations.
  """
  image_path = image_info.image_path
  image_name = os.path.basename(image_path)
  image_w, image_h = image_info.image_width, image_info.image_height
  image = Image.open(image_path).convert("RGB")

  # Count detections using the YOLO model
  class_counts = pg2_lib.count_yolo_detections(yolo_model, image_path, class_labels)
  detection_prompt = pg2_lib.create_detection_prompt(class_labels, class_counts)

  # Add <image> token to the prompt
  inputs_for_annotations = processor(text="<image>" + detection_prompt, images=image, return_tensors="pt").to(device)

  with torch.no_grad():
    output_annotations = model.generate(**inputs_for_annotations, max_new_tokens=500)

  generated_annotations_text = processor.decode(output_annotations[0], skip_special_tokens=True)
  if print_output:
    print(f"PaliGemma2 Output for {image_name}:\n{generated_annotations_text}\n")
  try:
    updated_bboxes = pg2_lib.parse_response_to_yolo_bbox(
      generated_annotations_text,
      image_w,
      image_h,
      class_labels,
      model_input_size=448,
      fail_on_error=skip_bad_annotation
    )
  except Exception as e:
    print(f"Error parsing PaliGemma2 output: {e}. Skipping annotation update for this image.")
    updated_bboxes = []

  if save_to_file:
    pg2_lib.save_annotations_to_file(image_name, dataset_path, updated_bboxes)

  return updated_bboxes

# The save_annotations_to_file function is in pg2_lib.py

def display_detection(
  image_info: pg2_lib.YoloImageInfo,
  bboxes: list[list[float]],
  class_labels: list[str],
  display_width: int,
) -> None:
  """
  Displays an image with drawn bounding boxes.

  Args:
    image_info (YoloImageInfo): Dataclass containing image path, width, height.
    bboxes (list of lists): A list of YOLO bounding box format annotations.
    class_labels (list[str]): A list of class names.
    display_width (int): Width to display the image.
  """
  image = Image.open(image_info.image_path).convert("RGB")
  image = pg2_lib.draw_bboxes_on_image(image, bboxes, class_labels)
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
sample_images_with_bboxes: list[tuple[pg2_lib.YoloImageInfo, list[list[float]]]] = []

for i, image_info in tqdm(enumerate(dataset_info[try_images[0]:try_images[1]])):
  bboxes = process_image_for_detection(
    image_info,
    yolo_model,
    class_labels,
    skip_bad_annotation=False,
    print_output=True,
    save_to_file=False
  )
  sample_images_with_bboxes.append((image_info, bboxes))

for image_info, bboxes in sample_images_with_bboxes:
  if bboxes:
    display_detection(image_info, bboxes, class_labels, display_width=600)

# NEXT CELL

dataset_info = pg2_lib.read_yolo_dataset(dataset_path)
assert dataset_info, "No dataset information?!"

for i, image_info in tqdm(enumerate(dataset_info)):
  bboxes = process_image_for_detection(
    image_info,
    yolo_model,
    class_labels,
    skip_bad_annotation=False,
    print_output=False
  )
  # Saving is done inside the function
