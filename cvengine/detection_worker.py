import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

# NOTE: logging is configured in config module
from config import ADDRESS, MAX_FRAME_QUEUE_SIZE, MODEL, PORT, SAVE_DETECTIONS, SCRIPTDIR, logging
import cvengine_pb2 as cve

_worker_model = None # Global variable for the model in worker processes

detection_path_tmpl = os.path.join(f"{SCRIPTDIR}", "results", "{}.jpg")
def save_detection(results: list[Results]):
  cv2.imwrite(detection_path_tmpl.format(round(time.time() * 1000)), results[0].plot())

def init_worker(model_path: str):
  """Initializes the YOLO model in each worker process."""
  global _worker_model
  try:
    logging.info(f"Worker process: Loading YOLOv12 model from {model_path}")
    _worker_model = YOLO(model_path)
    logging.info("Worker process: YOLOv12 model loaded.")
  except Exception as e:
    logging.error(f"Worker process: Error loading YOLOv12 model: {e}")
    raise

def detect_objects(image_np: np.ndarray) -> list[cve.DetectedObject]:
  """Standalone function for object detection"""
  global _worker_model
  if _worker_model is None:
    logging.error("Worker model not initialized in process.")
    return [] # Return empty list if model is not loaded

  results: list[Results] = _worker_model(image_np, verbose=False)
  if SAVE_DETECTIONS:
    save_detection(results)
  detected_objects: list[cve.DetectedObject] = []
  for r in results:
    if r.boxes is None or len(r.boxes.data) == 0:
      continue
    for *xyxy, conf, cls in r.boxes.data:
      class_name = _worker_model.names[int(cls)]
      bbox = cve.BoundingBox(
        xmin=float(xyxy[0]), ymin=float(xyxy[1]),
        xmax=float(xyxy[2]), ymax=float(xyxy[3])
      )
      detected_objects.append(
        cve.DetectedObject(
          class_name=class_name,
          confidence=float(conf),
          bbox=bbox
        )
      )
  return detected_objects