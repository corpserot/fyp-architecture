# Color analysis
## Week 2
- Worked on formulating a LAB-to-color-names dataset. Found dataset by Avishek Saha on Kaggle: <https://www.kaggle.com/datasets/avi1023/color-names>. Color values are transformed from RGB to LAB (pre-baking the data).
- Worked on a color distance scorer to rank how different a color is to another color (lower score is closer). this will be used to rank top 5 colors closest to the segmented clothing color. Uses the Delta E (ΔE or E*) formula.
  - https://alpolic-americas.com/blog/how-is-color-measured-calculating-delta-e/
  - https://www.viewsonic.com/library/creative-work/what-is-delta-e-and-why-is-it-important-for-color-accuracy/
  - https://www.lovibond.com/usa-en/PC/Colour-Values/XYZ-Tristimulus/XYZ-Tristimulus3

## Week 3
- Found the `colormath` library which conveniently includes many color operations and calculations. Previous implementation abandoned.
- Generated discrete points from a 3D gradient cylindrical volume in OKLch color space with the Lightness range 0.4-0.95 divided into 5 steps, Chroma (saturation) range 0.05 to 0.25 divided into 3 steps, Hue range 0-1 divided into 8 steps. Additionally, the greyscale gradient is divided into 8 steps. In total 128 colors to compare.
  - The palette serves as the visual reference for possible colors can be matched to clothes.
  - Abandoned self-implementation of the Delta E perceptual color difference implementation in favor of using the `colormath` library.
  - CIELAB is abandoned as I realized that CIELAB is older than OKLAB. For context, Lch is a flatter variant of LAB space. It is easier to work with as seen with components above. Meanwhile, LAB has Lightness, A and B component. A is the green-red intensity. B is the blue-yellow axis.

# CV Engine Server Architecture Version 1 (Week 4-5)

The CV Engine server is a high-performance, asynchronous gRPC service designed for real-time object detection. Produces bounding box of detected people with age+gender information. It leverages Python's `asyncio` for concurrent I/O and `ProcessPoolExecutor` for parallel CPU-bound tasks, ensuring efficient processing of multiple client (Main Backend Server) streams. Thusly supporting multiple Main Backend Servers if necessary. Utilizes Ultralytics YOLO, which automatically selects the best available device (GPU if CUDA is available, otherwise CPU).

This architecture will carry over to subsequent full implementation, and its current implementation allows testing the Age+Gender detection model.

## Week 5
- Worked on the basic form of the CV Engine. Implemented working gRPC methods + protobuf data structures.
  - Added capability to store the annotated results.
  - Created a "Test Client" that streams images to the CV Engine. In our architecture, this role would be fulfilled by the Main Backend Server. Implementation used here directly contributes to Main Backend Server.
  -  Added capability to stream video to CV Engine Test Client.

## Components

### 1. `cvengine.proto` (Protocol Definition)
Defines the gRPC service contract and data structures:
- `CVEngineService`: The main service with a single RPC method.
  - `DetectObjects` method: A bidirectional streaming RPC that takes a stream of `ImageFrame` messages and returns a stream of `DetectionResult` messages.
- `ImageFrame`: Contains raw image data (bytes) and a timestamp.
- `BoundingBox`: Contains `x_min`, `y_min`, `x_max`, `y_max`
- `DetectedObject`: Represents a detected object with class name, confidence and bounding box
- `DetectionResult`: Contains a list of `DetectedObject`s and a timestamp.

### 2. `server.py` (gRPC Server Entry Point)
Initializes and manages the gRPC server:
- Creates an `asyncio` gRPC server instance with a `ThreadPoolExecutor` for handling gRPC calls.
- Registers `CVEngineService` as the servicer.

### 3. `service.py` (CVEngineService Implementation)

The CVEngineServiceServicer implements a high-performance gRPC service for real-time object detection in a multi-client environment. It handles bidirectional streaming RPCs via the DetectObjects method, processing streams of ImageFrame messages from clients and yielding DetectionResult messages containing detected objects.

Core components include:
- A shared `asyncio.Queue` (`frame_queue`) buffering incoming `FrameWithClientId` objects from all clients.
- Client-specific `asyncio.Queue` mappings (detection_queues) for detection results.
- A `ProcessPoolExecutor` for offloading CPU-intensive detection tasks to worker processes, preserving the main event loop.
- An asynchronous `_processing_worker` task that dequeues frames, decodes images with OpenCV, runs detection via detect_objects in the executor, logs results, and enqueues DetectionResults.

The `DetectObjects` RPC method identifies clients via metadata, initializes a detection queue, spawns an input consumption task (`_consume_request_stream`), and yields results while logging throughput. It handles stream termination by sending disconnect signals (-2 timestamp) to the worker.

The `_consume_request_stream` task reads `ImageFrames` asynchronously, relieving pressure by dropping old frames if the queue exceeds `MAX_FRAME_QUEUE_SIZE` (sending empty results as drops), and forwards frames with client IDs. It propagates end-of-stream signals (-1 timestamp).

Shutdown coordinates graceful termination: signals clients (-1), halts the worker (-3), awaits tasks, shuts down the executor, and closes logging.

### 4. `detection_worker.py` (Object Detection Logic)

The `detection_worker.py` module handles the core computer vision logic for object detection, executing within separate processes managed by a `ProcessPoolExecutor`. It is responsible for loading the YOLO model, performing detections, and optionally saving annotated results.

Core components include:
- A global `_worker_model` that holds the loaded YOLO detection model, initialized once per process.
- `COLOR_PALETTE` and `_get_unique_color` for consistent annotation coloring.
- The `init_worker` function, which loads the YOLO model at the start of each worker process.
- The `detect_objects` function, which performs object detection on an input image, converts results to `cvengine_pb2.DetectedObject` format, and can optionally save annotated images via `save_detection`.

# CV Engine Rework (Week 6-7)
Attempt to change the CV Engine approach. For context, the pipeline according to my initial FYP1 proposal report is described below. All detection models are fine-tuned from pre-trained base YOLOv12. Bounding boxes can be created from both segmentation mask and polygon. Segmentation polygon can be created from segmentation mask.
1. Video frames, timestamps and camera ID are streamed from the Main Backend Server to the CV Engine via gRPC method `DetectObject`.
2. Age+Gender detection model produces bounding boxes detecting these 4 classes: `boy`, `girl`, `man`, `woman`.
3. In parallel, Clothing detection model produces segmentation mask detecting these 13 classes:  `short sleeve top`, `long sleeve top`, `short sleeve outwear`, `long sleeve outwear`, `vest`, `sling`, `shorts`, `trousers`, `skirt`, `short sleeve dress`, `long sleeve dress`, `vest dress` and `sling dress`.
4. Color analysis in OKLAB is performed on the segmentation to obtain major colors of the clothing matched following a predetermined palette.
5. OC-SORT, Observation-Centric SORT, tracks the bounding box produced by Age+Gender detection model. This produces object tracking IDs.
6. Together, the following data is sent back as the response to the Main Backend Server:
   - Timestamp and camera ID.
   - Object tracking IDs.
   - Bounding boxes (with confidence value) of people with age+gender information attached.
   - Segmentation polygon (with confidence value), and bounding box of their clothing.
   - Major colors (with difference value) of each clothing.

## Problems
The problem with step 3 (Clothing detection) is that it will likely misalign or disagree with the results from step 2 (Age+Gender detection) as the binding between the two is really weak and depends a lot on "coincidental agreement". 13 granular clothing classes makes it worse.

## Design Attempt 1 (Week 6)
Use an instance segmentation age+gender detection model, and a bounding box clothing model. The age+gender detection model doubles as people segmentation. The pipeline would look like this:
1. Video frames, timestamps and camera ID are streamed from the Main Backend Server to the CV Engine via gRPC method `DetectObject`.
2. Age+Gender detection model produces segmentation mask detecting the 4 classes.
3. Clothing detection model produces bounding boxes detecting the 13 classes.
4. Color analysis in OKLAB is performed on the segmentation to obtain major colors of the clothing matched following a predetermined palette.
5. (Unchanged) OC-SORT, Observation-Centric SORT, tracks the bounding box produced by Age+Gender detection model. This produces object tracking IDs.
6. Together, the following data is sent back as the response to the Main Backend Server:
   - Timestamp and camera ID.
   - Object tracking IDs.
   - Segmentation polygon (with confidence value), and bounding box of people with age+gender information attached.
   - Bounding boxes (with confidence value) of their clothing
   - Major colors (with difference value) of each clothing.

Effort was made into trying to salvage existing dataset by trying automated approaches to produce segmentation training data from bounding box utilizing pre-trained PaliGemma2. The quality of the training data however was massively reduced and this approached is abandoned.

In public, there are only small (<500 images) existing datasets adaptable to age+gender classes with varying quality, so this approach is abandoned too.

## Design Attempt 2 (Week 6-7)
Use 3 models: an instance segmentation people detection model, a bounding box age+gender detection model, and an instance segmentation clothing model. This approach has some benefits over Attempt 1:

- The people detection model provides a strong ground truth to both age+gender and segmentation clothing model. A YOLOv12 large (L) model variant trained for 300 epochs using a narrowed COCO dataset to only detect `person` class was found with the following metrics at https://huggingface.co/RyanJames/yolo12l-person-seg
  - Box mAP50-95 (COCO): 0.642
  - Box mAP50 (COCO): 0.851
  - Mask mAP50-95: 0.537
  - Mask mAP50: 0.837
  - Box Precision: 0.840
  - Box Recall: 0.759
  - Mask Precision: 0.843
  - Mask Recall: 0.748
- Both age+gender detection model and clothing model are allowed to fail. For example, if the age+gender detection model does not detect anything over a person, then there is no age+gender data for that frame.
- This also allows skipping trying to detect age+gender and clothing information entirely, depending on how visually small people are in the frame. This helps reduce CV Engine load.

The pipeline now looks like this:
1. Video frames, timestamps and camera ID are streamed from the Main Backend Server to the CV Engine via gRPC method `DetectObject`.
2. Person detection model produces segmentation mask detecting the only class, `person`.
3. OC-SORT tracks the segmentation mask produced by the Person detection model. This produces object tracking IDs.
4. Periodically update the age+gender association of tracking IDs. Check if there is any people visually large enough to try and detect age+gender. If too small, skip this step. Otherwise, Age+Gender detection model produces bounding boxes from the entire frame, detecting the 4 classes.
   - IoU matching is used to associate results to each person.
4. Periodically update the clothing association of tracking IDs. Check if there is any people visually large enough to try and detect clothing. If too small, skip this step. Otherwise, clothing detection model produces segmentation from the entire frame, detecting the 13 classes.
   - IoU matching is used to associate results to each person.
   - The 13 classes are grouped to these categories: `top`, `bottom` and `dress`.
5. Together, the following data is sent back as the response to the Main Backend Server:
   - Timestamp and camera ID.
   - Object tracking IDs.
   - Segmentation polygon (with confidence value) of each person.
   - If detected, age+gender information and confidence value.
   - If detected, top clothing class/type, confidence value, its major colors and color difference values.
