import csv
import logging
import os
import time
from typing import Iterable

import cvengine_pb2 as cve

class CSVLogger:
  def __init__(self, script_dir: str, filename: str = "detection_results.csv"):
    self.detection_csv_path = os.path.join(script_dir, filename)
    self.csv_file = None
    self.csv_writer = None
    self._initialize_csv()

  def _open_csv_file(self):
    """Opens the CSV file and initializes the writer."""
    file_exists = os.path.exists(self.detection_csv_path)
    self.csv_file = open(self.detection_csv_path, 'a', newline='')
    self.csv_writer = csv.writer(self.csv_file)
    if not file_exists:
      self.csv_writer.writerow(['timestamp', 'class_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])
    return file_exists

  def _initialize_csv(self):
    """Initializes the CSV file and writer."""
    self._open_csv_file()
    logging.info(f"Initialized CSV file: {self.detection_csv_path}")

  def _reinitialize_csv(self, retries: int = 2):
    """Attempts to reinitialize the CSV writer with retries."""
    for attempt in range(retries):
      logging.warning(f"Attempting to reinitialize CSV writer (Attempt {attempt + 1}/{retries})...")
      try:
        if self.csv_file:
          self.csv_file.close()
        self._open_csv_file()
        logging.info("CSV writer reinitialized successfully.")
        return True
      except Exception as e:
        logging.error(f"Error reinitializing CSV writer: {e}")
        time.sleep(0.1)  # Small delay before retrying
    logging.error("Failed to reinitialize CSV writer after multiple attempts.")
    return False

  def log_detection(self, timestamp: int, detected_objects: Iterable[cve.DetectedObject]):
    if not self.csv_writer:
      logging.error("Attempted to log detection, but CSV writer is unavailable.")
      if not self._reinitialize_csv():
        return  # Give up if reinitialization fails

    if not detected_objects:
      if self.csv_writer:
        self.csv_writer.writerow([timestamp, 'no_objects', '', '', '', '', ''])
      return

    for obj in detected_objects:
      if self.csv_writer:
        self.csv_writer.writerow([
          timestamp,
          obj.class_name,
          obj.confidence,
          obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax
        ])

    if self.csv_file:
      self.csv_file.flush()  # Ensure data is written to disk immediately

  def close(self):
    if self.csv_file:
      self.csv_file.close()
      logging.info(f"Closed CSV file: {self.detection_csv_path}")
