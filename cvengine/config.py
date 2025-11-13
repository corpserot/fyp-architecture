import os
import logging

SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
MODEL = 'detect-n.pt' # Choose from models/
ADDRESS = '127.0.0.1'
PORT = '5501'
MAX_FRAME_QUEUE_SIZE = 5 # Max frames to buffer before skipping

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s | %(levelname)s | %(message)s',
  handlers=[
    logging.FileHandler(f'{SCRIPTDIR}/cvengine-server-log.txt', mode='a'),
    logging.StreamHandler() # Also log to console
  ]
)