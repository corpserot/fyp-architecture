import asyncio
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
import time
from typing import AsyncIterator

import cv2
import grpc
from grpc.aio import Metadata
import numpy as np

# NOTE: logging is configured in config module
from config import ADDRESS, MAX_FRAME_QUEUE_SIZE, MODEL, PORT, SCRIPTDIR, logging
from csv_logger import CSVLogger
import cvengine_pb2 as cve
import cvengine_pb2_grpc as cve_grpc
from detection_worker import init_worker, detect_objects

@dataclass
class FrameWithClientId:
  frame: cve.ImageFrame
  client_id: str

class CVEngineService(cve_grpc.CVEngineServiceServicer):
  def __init__(self):
    self.frame_queue = asyncio.Queue[FrameWithClientId]()
    self.detection_queues: dict[str, asyncio.Queue[cve.DetectionResult]] = {}

    self.csv_logger = CSVLogger(SCRIPTDIR)

    self.executor = ProcessPoolExecutor(
        max_workers=os.cpu_count() or 4,
        initializer=init_worker,
        initargs=(f"{SCRIPTDIR}/models/{MODEL}",)
    )
    self.processing_task = asyncio.create_task(self._processing_worker())

  async def shutdown(self):
    """Gracefully shuts down the service."""
    logging.info("Shutting down CVEngineServiceServicer...")

    # Signal all active client streams to terminate gracefully
    logging.info(f"Signaling {len(self.detection_queues)} active clients to disconnect.")
    for client_id, queue in self.detection_queues.items():
      try:
        # A timestamp of -1 signals the end of the stream to the client
        await queue.put(cve.DetectionResult(timestamp=-1))
      except Exception as e:
        logging.warning(f"Error sending shutdown signal to client {client_id}: {e}")

    # Signal the processing worker to shut down after processing remaining frames
    logging.info("Signaling processing worker to shut down.")
    await self.frame_queue.put(FrameWithClientId(frame=cve.ImageFrame(timestamp=-3), client_id="shutdown"))

    # Wait for the processing task to finish gracefully
    if self.processing_task:
      await self.processing_task
      logging.info("Processing worker task finished.")

    # Shut down the process pool executor
    if self.executor:
      self.executor.shutdown(wait=True)
      logging.info("ProcessPoolExecutor shut down.")

    # Close the CSV logger
    self.csv_logger.close()
    logging.info("CSVLogger closed.")

  async def _processing_worker(self):
    loop = asyncio.get_running_loop()
    while True:
      frame_with_id = await self.frame_queue.get()
      current_frame = frame_with_id.frame
      client_id = frame_with_id.client_id

      # check early since detection_queue might not be available
      if current_frame.timestamp == -3:
        logging.info("Worker: Received shutdown signal. Exiting.")
        break # Exit the loop to allow graceful shutdown

      detection_queue = self.detection_queues.get(client_id)
      if not detection_queue:
        logging.warning(f"Worker: No result queue found for client {client_id}. Skipping frame {current_frame.timestamp}.")
        continue

      if current_frame.timestamp == -1:
        logging.info(f"Worker: Received end of stream signal for client {client_id}. Propagating sentinel.")
        await detection_queue.put(cve.DetectionResult(timestamp=-1))
        continue
      elif current_frame.timestamp == -2:
        logging.info(f"Worker: Received client disconnect signal for client {client_id}. Removing detection queue.")
        if client_id in self.detection_queues:
          del self.detection_queues[client_id]
        continue

      detected_objects: list[cve.DetectedObject] = []
      try:
        np_arr = np.frombuffer(current_frame.image_data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image_np is None:
          logging.warning(f"Worker: Could not decode image frame with timestamp {current_frame.timestamp}. Sending empty detection.")
        else:
          # Call the standalone detect_objects
          detected_objects = await loop.run_in_executor(
              self.executor,
              detect_objects,
              image_np
          )
        self.csv_logger.log_detection(current_frame.timestamp, detected_objects)

        detection_result = cve.DetectionResult(
          objects=detected_objects,
          timestamp=current_frame.timestamp
        )
        await detection_queue.put(detection_result)
      except Exception as e:
        logging.error(f"Worker: Error processing image frame {current_frame.timestamp} for client {client_id}: {e}. Sending empty detection.")
        await detection_queue.put(cve.DetectionResult(timestamp=current_frame.timestamp, objects=[]))

  async def _get_client_id(self, context: grpc.aio.ServicerContext) -> str:
    # NOTE: there's no correct way to solve this because of the incorrect
    # invocation_metadata type hint. we forcefully cast it.
    metadata: Metadata = context.invocation_metadata() # type: ignore
    client_id = None
    for key, value in metadata:
      if key == "client_id":
        client_id = value
        break

    if client_id is None:
      logging.error("Client ID not found in invocation metadata.")
      raise grpc.aio.AioRpcError(
        grpc.StatusCode.UNAUTHENTICATED,
        details="Client ID is required.",
        initial_metadata=Metadata(),
        trailing_metadata=Metadata()
      )
    return str(client_id)

  async def DetectObjects(self,
    request_iterator: AsyncIterator[cve.ImageFrame],
    context: grpc.aio.ServicerContext
  ) -> AsyncIterator[cve.DetectionResult]:

    client_id = await self._get_client_id(context)
    logging.info(f"DetectObjects stream started for client: {client_id}")
    detection_queue = asyncio.Queue[cve.DetectionResult]()
    self.detection_queues[client_id] = detection_queue

    input_task = asyncio.create_task(self._consume_request_stream(request_iterator, client_id))

    detection_results_sent_in_interval = 0
    last_log_time_sent = time.time()

    try:
      while True:
        detection_result = await detection_queue.get()
        if detection_result.timestamp == -1:
          logging.info(f"DetectObjects: Received end of stream signal for client: {client_id}.")
          break

        yield detection_result

        detection_results_sent_in_interval += 1
        current_time = time.time()
        if current_time - last_log_time_sent >= 1.0:
          logging.info(f"Server: Sent {detection_results_sent_in_interval} detection results to client {client_id} in the last second.")
          detection_results_sent_in_interval = 0
          last_log_time_sent = current_time

    except Exception as e:
      logging.error(f"DetectObjects for client {client_id}: Error yielding detection result: {e}")
    finally:
      input_task.cancel()
      await asyncio.gather(input_task, return_exceptions=True)
      # Signal the processing worker that this client's stream has ended
      await self.frame_queue.put(FrameWithClientId(frame=cve.ImageFrame(timestamp=-2), client_id=client_id))
      logging.info(f"DetectObjects stream finished for client: {client_id}. Sent client disconnect signal to worker.")

  async def _consume_request_stream(self,
    request_iterator: AsyncIterator[cve.ImageFrame],
    client_id: str
  ):
    """Consumes the incoming request stream and adds frames to the queue."""
    frames_received_in_interval = 0
    dropped_frames_in_interval = 0
    last_log_time_received = time.time()

    try:
      async for image_frame in request_iterator:
        logging.debug(f"Server: Raw frame {image_frame.timestamp} received by gRPC stream for client {client_id}.")

        frames_received_in_interval += 1
        current_time = time.time()
        if current_time - last_log_time_received >= 1.0:
          logging.info(f"Server: Received {frames_received_in_interval} frames, dropped {dropped_frames_in_interval} from client {client_id} in the last second.")
          frames_received_in_interval = 0
          dropped_frames_in_interval = 0
          last_log_time_received = current_time

        if image_frame.timestamp == -1:
          logging.info(f"DetectObjects: Client {client_id} signaled end of stream.")
          await self.frame_queue.put(FrameWithClientId(frame=image_frame, client_id=client_id))
          break

        if self.frame_queue.qsize() >= MAX_FRAME_QUEUE_SIZE:
          dropped_frame = await self.frame_queue.get()
          dropped_client_id = dropped_frame.client_id
          dropped_timestamp = dropped_frame.frame.timestamp
          dropped_frames_in_interval += 1

          dropped_detection_queue = self.detection_queues.get(dropped_client_id)
          if dropped_detection_queue:
            await dropped_detection_queue.put(cve.DetectionResult(timestamp=dropped_timestamp, objects=[]))
          else:
            logging.error(f"DetectObjects: No result queue found for dropped frame's client {dropped_client_id}. Cannot send empty detection.")

        await self.frame_queue.put(FrameWithClientId(frame=image_frame, client_id=client_id))
    except asyncio.CancelledError:
      logging.info(f"DetectObjects for client {client_id}: Input stream consumption cancelled.")
    except Exception as e:
      logging.error(f"DetectObjects for client {client_id}: Error consuming request stream: {e}")
