import asyncio
import logging
import os
import time
import uuid

import cv2
import grpc
import numpy as np

# running from the same directory
import cvengine_pb2
import cvengine_pb2_grpc

SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))

ADDRESS = '127.0.0.1'
PORT = '5501'

# how many cameras to simulate
CAMERA_COUNT = 1
# how much frames to send per second
FPS = 5 * CAMERA_COUNT
# seconds, negative means forever
DURATION = -1
# positive means exact count of frames, negative means multiple of all test frames
FRAME_COUNT = -1
# whether to loop or end the stream
FRAME_LOOP = False

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s | %(levelname)s | %(message)s',
  handlers=[
    logging.FileHandler(f'{SCRIPTDIR}/cvengine-client-log.txt', mode='a'),
    logging.StreamHandler() # Also log to console
  ]
)

async def generate_image_frames(
  image_dir: str,
  fps: int = FPS,
  duration: float | None = None,
  max_frame: int = FRAME_COUNT,
  loop_frames: bool = FRAME_LOOP
):

  image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
  if not image_files:
    logging.warning(f"No image files found in {image_dir}. Please add some images.")
    return

  image_files.sort() # Ensure consistent order
  num_images = len(image_files)
  logging.info(f"Found {num_images} images in {image_dir}.")

  # Normalize duration
  if duration is not None and duration < 0:
    duration = None;

  # Normalize max frames to send
  if max_frame < 0:
    max_frame = abs(max_frame) * num_images
  elif max_frame > num_images:
    logging.warning(f"Trying to send {max_frame} however. Will loop back to the start...")

  logging.info(f"Attempting to send at most {max_frame} frames {"forever" if duration is None else f"for {duration} seconds"} with {fps} fps...")

  start_time = time.time()
  frame_count = 0
  last_log_time = time.time()
  frames_sent_in_interval = 0

  while True:
    if duration is not None and (time.time() - start_time) >= duration:
      break

    if not loop_frames and frame_count > max_frame:
      break

    image_path = os.path.join(image_dir, image_files[frame_count % num_images])
    image_np = cv2.imread(image_path)
    frame_count += 1
    frames_sent_in_interval += 1

    if image_np is None:
      logging.warning(f"Could not read image: {image_path}. Skipping.")
      continue

    # Encode image to bytes
    _, img_encoded = cv2.imencode('.jpg', image_np)
    image_data = img_encoded.tobytes()

    timestamp = int(time.time() * 1000) # Milliseconds
    yield cvengine_pb2.ImageFrame(image_data=image_data, timestamp=timestamp)
    del img_encoded, image_data

    current_time = time.time()
    if current_time - last_log_time >= 1.0:
      logging.info(f"Client: Sent {frames_sent_in_interval} frames in the last second.")
      frames_sent_in_interval = 0
      last_log_time = current_time

    # Calculate sleep time to maintain desired FPS
    next_frame_time = start_time + (frame_count / fps)
    if next_frame_time - time.time() > 0:
      await asyncio.sleep(next_frame_time - time.time())

  # Send a sentinel value to signal the end of the stream
  logging.info("Client: Sending end of stream signal.")
  yield cvengine_pb2.ImageFrame(timestamp=-1)

async def run_client():
  client_id = str(uuid.uuid4()) # Generate a unique client ID
  async with grpc.aio.insecure_channel(f'{ADDRESS}:{PORT}') as channel:
    stub = cvengine_pb2_grpc.CVEngineServiceStub(channel)
    logging.info(f"Client {client_id}: gRPC client connected to {ADDRESS}:{PORT}")

    image_directory = os.path.join(SCRIPTDIR, "test_images")
    logging.info(f"Client {client_id}: Started streaming images for detection.")
    response_stream = stub.DetectObjects(generate_image_frames(image_directory), metadata=[('client_id', client_id)])
    frames_received_in_interval = 0
    last_log_time = time.time()

    try:
      async for detection_result in response_stream:
        if detection_result.timestamp == -1:
          logging.info(f"Client {client_id}: Received end of stream signal.")
          break
        frames_received_in_interval += 1
        current_time = time.time()
        if current_time - last_log_time >= 1.0:
          logging.info(f"Client {client_id}: Received {frames_received_in_interval} detection results in the last second.")
          frames_received_in_interval = 0
          last_log_time = current_time
    except asyncio.CancelledError:
      logging.info(f"Client {client_id}: Stream consumption cancelled.")
    except grpc.aio.AioRpcError as e:
      logging.error(f"Client {client_id}: gRPC error: {e.code()} - {e.details()}")
    except Exception as e:
      logging.error(f"Client {client_id}: An unexpected error occurred: {e}")
    finally:
      logging.info(f"Client {client_id}: Finished streaming images for detection.")

if __name__ == '__main__':
  try:
    asyncio.run(run_client())
  except KeyboardInterrupt:
    logging.info("Client interrupted by user. Shutting down...")
  except Exception as e:
    logging.error(f"An unexpected error occurred: {e}")
