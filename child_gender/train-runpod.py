# BEGIN IPYTHON
# END IPYTHON

# BEGIN MARKDOWN
# ## 1. Install dependencies
# END MARKDOWN

import os
import locale
locale.getpreferredencoding = lambda: "UTF-8"

os.system('nvidia-smi') # BANG: !nvidia-smi

# Setup paths
os.chdir('/workspace') # MAGIC: %cd /workspace
home = os.getcwd()
dataset_path = "/workspace/dataset"

os.system('pip install -q ultralytics gdown') # MAGIC: %pip install -q ultralytics gdown

# NEXT CELL

os.system(f"gdown --fuzzy 'https://drive.google.com/file/d/11tsGuerUAMnfYuOSgn-IS5iRbIPjQA2j/view?usp=sharing' -O {home}/dataset.zip") # BANG: !gdown --fuzzy 'https://drive.google.com/file/d/11tsGuerUAMnfYuOSgn-IS5iRbIPjQA2j/view?usp=sharing' -O {home}/dataset.zip
os.system(f"mkdir -p {home}/dataset") # BANG: !mkdir -p {home}/dataset
os.system(f"rm -rf {home}/dataset") # BANG: !rm -rf {home}/dataset
os.system(f"unzip -q {home}/dataset.zip -d {home}/dataset") # BANG: !unzip -q {home}/dataset.zip -d {home}/dataset

# BEGIN MARKDOWN
# ## 2. Setup Ultralytics
# END MARKDOWN

import ultralytics
from ultralytics import YOLO

ultralytics.checks()

# NEXT CELL

dataset_path = f'/{home}/dataset'
runs_path = f'{home}/runs/detect'

# BEGIN MARKDOWN
# ## 3. Fine-tune YOLOv12 model on a Custom Dataset
# END MARKDOWN

os.system(f"rm -rf {runs_path}") # BANG: !rm -rf {runs_path}

# NEXT CELL

# choose one:

model_names = (
  "yolo12n.pt",
  "yolo12s.pt",
  "yolo12m.pt",
  # "yolo12l.pt",
  # "yolo12x.pt"
)

for nr, model_name in enumerate(model_names):
  train_nr = str(nr+1)
  if train_nr == "1":
    train_nr = ''
  if nr < -1: continue
  model = YOLO(model_name)
  results = model.train(
    data=f'{dataset_path}/data.yaml',
    batch=64, # batch size
    cos_lr=True, # adjusts the learning rate following cosine decay
    # use default learning rate (0.005)
    epochs=100, # train for at most N epochs
    patience=20, # stops if there's N epochs with no improvments
    # use default weight decay (0.005)
    # use default warmup epochs (3) and LR bias (0)

    deterministic=False, # small performance boost
    optimizer="AdamW",

    # NOTE: see online visualizations of the augmentations too
    mosaic=0.2, # mosaic augmentation chance
    close_mosaic=30, # stops mosaic augmentation for last N epochs
    copy_paste=0.2, # copy paste augmentation chance
    erasing=0.2, # erasing augmentation chance

    hsv_h=0.015, hsv_s=0.3, hsv_v=0.3, # max hue, saturation, value shift range
    scale=0.15, # max scaling range
    translate=0.1, # max translation range
    degrees=15, # max rotation range
  )

  best_model_path = f'{runs_path}/train{train_nr}/weights/best.pt'
  best_model = YOLO(best_model_path)

  results = best_model.val()
  results = best_model.val(split='test')
  results = best_model.predict(source = f"{dataset_path}/test/images", save = True)