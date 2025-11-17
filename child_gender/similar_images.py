import os
import shutil
import numpy as np
from typing import List, Tuple, Dict, Iterator
from dataclasses import dataclass, field
from collections import OrderedDict

import cv2
from PIL import Image
import imagehash
from imagehash import ImageHash
from skimage.metrics import structural_similarity as ski_ssim
from tqdm import tqdm

# This removes perceptually similar images

SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPTDIR, 'dataset_raw')
TMP_PATH = os.path.join(SCRIPTDIR, 'dataset_tmp')
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
COPY_IMAGES = False
DELETE_SIMILARS = False
# None for all
IMAGE_RANGE_TO_PROCESS = None #[512, 2048]

# Constants for split names and file extensions
SPLIT_NAMES = ['train', 'valid', 'test']
LABEL_FILE_EXT = '.txt'

SIMILARS_FILE = os.path.join(SCRIPTDIR, 'similar.txt')

# Maximum Hamming distance for pHash for images to be considered similar
PHASH_HAMMING_DISTANCE_THRESHOLD = 3
# If pHash distance is above this, images are considered too different to proceed with further checks.
PHASH_TOO_DIFFERENT_THRESHOLD = 15
# Threshold for pHash distance to consider images "potentially similar" for ORB+RANSAC check.
# If pHash distance is between PHASH_HAMMING_DISTANCE_THRESHOLD and PHASH_POTENTIALLY_SIMILAR_THRESHOLD,
# ORB+RANSAC may be performed, if SSIM agrees. Otherwise, it's either definitely similar or too different.
PHASH_POTENTIALLY_SIMILAR_THRESHOLD = 10

# Minimum SSIM score for images to be considered similar (0.0 to 1.0)
SSIM_THRESHOLD = 0.9 # Increased to make SSIM stricter
# Threshold for SSIM score to consider images "potentially similar" for ORB+RANSAC check.
# If SSIM score is between SSIM_THRESHOLD and SSIM_POTENTIALLY_SIMILAR_THRESHOLD,
# ORB+RANSAC may be performed, if pHash agrees. Otherwise, it's either definitely similar or too different.
SSIM_POTENTIALLY_SIMILAR_THRESHOLD = 0.70 # Lowered to capture more subtle similarities for ORB

# ORB (Oriented FAST and Rotated BRIEF) parameters
# Maximum number of strongest features (key points) to retain.
ORB_FEATURES = 300
# Reciprocal scale images are shrunk at in finding features (pyramid). Must be greater than 1.
ORB_SCALE_FACTOR = 1.4
# The number of pyramid levels.
ORB_N_LEVELS = 3

# FLANN (Fast Library for Approximate Nearest Neighbors) parameters for descriptor matching
FLANN_INDEX_LSH = 6 # Use Locality Sensitive Hashing method
FLANN_INDEX_PARAMS = dict(
  algorithm = FLANN_INDEX_LSH,
  table_number = 6,      # suggested range: 6-12 (less -> more greedy)
  key_size = 12,         # suggested range: 12-20
  multi_probe_level = 1) # suggested range: 1-2 (less -> more greedy)

# RANSAC (Random Sample Consensus) parameters for geometric verification
# Maximum allowed reprojection error to treat a point pair as an inlier.
RANSAC_REPROJECTION_THRESHOLD = 2.0 # less -> more lenient similarity threshold
# Minimum number of inliers required to consider images geometrically similar.
RANSAC_MIN_INLIERS = 80 # less -> more strict similarity threshold
# Minimum ratio of inliers to total matches for geometric similarity.
RANSAC_INLIER_RATIO_THRESHOLD = 0.90 # less -> more strict similarity threshold

# Cache for fully loaded image details to avoid repeated disk I/O
# Stores ImageInfo objects with full details (cv2_img, pil_img, etc.)
IMAGE_CACHE_SIZE = 1024 # Adjust based on memory constraints and performance needs

@dataclass
class ImageInfo:
  """
  Stores pre-calculated information for an image to avoid redundant loading and processing.
  """
  path: str
  phash_val: ImageHash
  cv2_img: np.ndarray | None = field(default=None, repr=False)
  pil_img: Image.Image | None = field(default=None, repr=False)
  cv2_img_gray: np.ndarray | None = field(default=None, repr=False)
  cv2_img_downscaled_gray: np.ndarray | None = field(default=None, repr=False) # For ORB
  orb_keypoints: Tuple[cv2.KeyPoint, ...] | None = field(default=None, repr=False)
  orb_descriptors: np.ndarray | None = field(default=None, repr=False)
  similarity_reason: str = field(default='none', repr=False)

  def close(self):
    """Explicitly closes the PIL image to release file handles."""
    if self.pil_img:
      self.pil_img.close()
      self.pil_img = None

image_cache: OrderedDict[str, ImageInfo] = OrderedDict()

# Initialize ORB detector globally
orb_detector: cv2.ORB = cv2.ORB_create( # type: ignore
  nfeatures=ORB_FEATURES,
  scaleFactor=ORB_SCALE_FACTOR,
  nlevels=ORB_N_LEVELS
)

# Initialize FLANN matcher globally
flann_matcher: cv2.FlannBasedMatcher = cv2.FlannBasedMatcher(FLANN_INDEX_PARAMS, {}) # type: ignore

class DisjointSetUnion:
  """
  A Disjoint Set Union (DSU) data structure for efficiently managing sets of elements.
  """
  def __init__(self, elements: List[str]):
    """
    Initializes the DSU with a list of elements. Each element starts
    in its own set.

    Args:
      elements (List[str]): A list of unique elements to initialize the DSU with.
    """
    self.parent: Dict[str, str] = {element: element for element in elements}
    self.rank: Dict[str, int] = {element: 0 for element in elements}
    self.num_sets = len(elements)

  def find(self, element: str) -> str:
    """
    Finds the representative (root) of the set containing the given element.
    Performs path compression for optimization.

    Args:
      element (str): The element to find the representative for.

    Returns:
      str: The representative of the element's set.
    """
    if self.parent[element] == element:
      return element
    self.parent[element] = self.find(self.parent[element])
    return self.parent[element]

  def union(self, element1: str, element2: str) -> bool:
    """
    Unites the sets containing element1 and element2.
    Performs union by rank for optimization.

    Args:
      element1 (str): The first element.
      element2 (str): The second element.

    Returns:
      bool: True if the sets were united, False if they were already in the same set.
    """
    root1 = self.find(element1)
    root2 = self.find(element2)

    if root1 != root2:
      if self.rank[root1] < self.rank[root2]:
        self.parent[root1] = root2
      elif self.rank[root1] > self.rank[root2]:
        self.parent[root2] = root1
      else:
        self.parent[root2] = root1
        self.rank[root1] += 1
      self.num_sets -= 1
      return True
    return False

  def get_groups(self) -> List[List[str]]:
    """
    Returns a list of lists, where each inner list represents a disjoint set (group).
    """
    groups: Dict[str, List[str]] = {}
    for element in self.parent:
      root = self.find(element)
      if root not in groups:
        groups[root] = []
      groups[root].append(element)
    return list(groups.values())

def load_and_process_image(img_path: str) -> ImageInfo | None:
  """
  Loads an image and pre-calculates only the perceptual hash.
  Other features are loaded lazily.

  Args:
    img_path (str): The path to the image file.

  Returns:
    ImageInfo | None: An ImageInfo object with path and phash, or None if an error occurs.
  """
  try:
    pil_img = Image.open(img_path)
    phash_val = imagehash.phash(pil_img)
    return ImageInfo(path=img_path, phash_val=phash_val)
  except Exception as e:
    print(f"Error processing image {img_path}: {e}")
    return None

def load_full_image_details(img_info: ImageInfo):
  """
  Loads the full image details (cv2_img, pil_img, cv2_img_gray)
  into an existing ImageInfo object, utilizing an LRU cache.

  Args:
    img_info (ImageInfo): The ImageInfo object to populate.
  """
  if img_info.path in image_cache:
    cached_info = image_cache.pop(img_info.path) # Move to end of LRU
    image_cache[img_info.path] = cached_info
    # Copy loaded details to the current img_info object
    img_info.cv2_img = cached_info.cv2_img
    img_info.pil_img = cached_info.pil_img
    img_info.cv2_img_gray = cached_info.cv2_img_gray
    img_info.cv2_img_downscaled_gray = cached_info.cv2_img_downscaled_gray
    img_info.orb_keypoints = cached_info.orb_keypoints
    img_info.orb_descriptors = cached_info.orb_descriptors
    return

  try:
    cv2_img = cv2.imread(img_info.path)
    if cv2_img is None:
      print(f"Warning: Could not load image with OpenCV: {img_info.path}")
      return

    pil_img = Image.open(img_info.path)
    cv2_img_gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

    # Downscale for ORB processing
    h, w = cv2_img_gray.shape
    downscale_factor = 2 # Example: downscale to half size
    downscaled_h, downscaled_w = h // downscale_factor, w // downscale_factor
    cv2_img_downscaled_gray = cv2.resize(cv2_img_gray, (downscaled_w, downscaled_h), interpolation=cv2.INTER_AREA)

    # Detect ORB keypoints and compute descriptors
    kp, des = orb_detector.detectAndCompute(cv2_img_downscaled_gray, None) # type: ignore

    # Populate the current img_info object
    img_info.cv2_img = cv2_img
    img_info.pil_img = pil_img
    img_info.cv2_img_gray = cv2_img_gray
    img_info.cv2_img_downscaled_gray = cv2_img_downscaled_gray
    img_info.orb_keypoints = tuple(kp) # Store as tuple for immutability
    img_info.orb_descriptors = des

    # Add to cache
    image_cache[img_info.path] = img_info
    if len(image_cache) > IMAGE_CACHE_SIZE:
      # Remove the least recently used item
      oldest_img_info = image_cache.popitem(last=False)[1]
      # Explicitly close the PIL image and clear references in the oldest item to aid garbage collection
      oldest_img_info.close() # Close the PIL image
      oldest_img_info.cv2_img = None
      oldest_img_info.cv2_img_gray = None
      oldest_img_info.cv2_img_downscaled_gray = None
      oldest_img_info.orb_keypoints = None
      oldest_img_info.orb_descriptors = None

  except Exception as e:
    print(f"Error loading full details for image {img_info.path}: {e}")
    # Clear partially loaded data and close PIL image to avoid inconsistent state
    img_info.close() # Close the PIL image
    img_info.cv2_img = None
    img_info.cv2_img_gray = None
    img_info.cv2_img_downscaled_gray = None
    img_info.orb_keypoints = None
    img_info.orb_descriptors = None

def setup_tmp_directories(tmp_path: str):
  """
  Sets up temporary directories for images and labels.
  """
  shutil.rmtree(tmp_path, ignore_errors=True)
  os.makedirs(f'{tmp_path}/images', exist_ok=True)
  os.makedirs(f'{tmp_path}/labels', exist_ok=True)

def copy_dataset_split(
  src_path: str,
  split_name: str,
  tmp_images_path: str,
  tmp_labels_path: str,
):
  """
  Copies images and labels for a given split into temporary directories.
  """
  split_src_images_path = os.path.join(src_path, split_name, 'images')
  split_src_labels_path = os.path.join(src_path, split_name, 'labels')

  for filename in tqdm(os.listdir(split_src_images_path), desc=f"Copying images in {os.path.join(os.path.basename(src_path), split_name)}"):
    if filename.lower().endswith(IMAGE_EXTS):
      shutil.copy(
        os.path.join(split_src_images_path, filename),
        os.path.join(tmp_images_path, filename)
      )
      label_filename = os.path.splitext(filename)[0] + LABEL_FILE_EXT
      if os.path.exists(os.path.join(split_src_labels_path, label_filename)):
        shutil.copy(
          os.path.join(split_src_labels_path, label_filename),
          os.path.join(tmp_labels_path, label_filename)
        )

def calculate_ssim(
  img1_gray: np.ndarray,
  img2_gray: np.ndarray,
) -> bool:
  """
  Calculates the Structural Similarity Index (SSIM) between two images.
  """
  ssim_result = ski_ssim(img1_gray, img2_gray, full=True)
  return ssim_result[0] > SSIM_THRESHOLD

def calculate_phash_distance(
  h1: ImageHash,
  h2: ImageHash,
) -> bool:
  """
  Compares two images using perceptual hashing (pHash).
  """
  return (h1 - h2) < PHASH_HAMMING_DISTANCE_THRESHOLD

def are_images_similar(
  img_info1: ImageInfo,
  img_info2: ImageInfo,
) -> Tuple[bool, str]:
  """
  Determines if two images are similar using a combination of methods,
  prioritizing more reliable checks first.

  Returns:
    Tuple[bool, str]: A tuple where the first element is True if images are similar,
                      and the second element is a string indicating the reason for similarity
                      ('phash', 'ssim', 'orb', or 'none').
  """
  # 1. Perceptual Hashing (pHash) - this is already loaded
  phash_dist = img_info1.phash_val - img_info2.phash_val
  if phash_dist < PHASH_HAMMING_DISTANCE_THRESHOLD:
    return True, 'phash' # Definitely similar by pHash

  # If pHash indicates images are too different, skip further checks
  if phash_dist > PHASH_TOO_DIFFERENT_THRESHOLD:
    return False, 'none'

  # Load full image details for further comparisons
  load_full_image_details(img_info1)
  load_full_image_details(img_info2)

  # If full details could not be loaded for either image, they cannot be compared further
  if img_info1.cv2_img is None or img_info2.cv2_img is None:
    return False, 'none'

  # Scale images to the same dimensions for SSIM calculation
  h1, w1, _ = img_info1.cv2_img.shape
  h2, w2, _ = img_info2.cv2_img.shape

  # Determine target dimensions (smaller of the two)
  target_h = min(h1, h2)
  target_w = min(w1, w2)

  # Resize both images to the target dimensions
  resized_img1 = cv2.resize(img_info1.cv2_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
  resized_img2 = cv2.resize(img_info2.cv2_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

  # 2. Structural Similarity (SSIM) - using color images with multichannel=True
  ssim_result = ski_ssim(resized_img1, resized_img2, full=True, channel_axis=-1)
  ssim_score = ssim_result[0]
  if ssim_score > SSIM_THRESHOLD:
    return True, 'ssim' # Definitely similar by SSIM

  # If pHash and SSIM indicate potential similarity, proceed with ORB+RANSAC
  if (PHASH_HAMMING_DISTANCE_THRESHOLD <= phash_dist <= PHASH_POTENTIALLY_SIMILAR_THRESHOLD) and \
     (SSIM_POTENTIALLY_SIMILAR_THRESHOLD <= ssim_score < SSIM_THRESHOLD):
    # 3. ORB + Geometric Verification
    if orb_geometric_verification(img_info1, img_info2):
      return True, 'orb'

  return False, 'none'

def find_orb_matches(
  img_info1: ImageInfo,
  img_info2: ImageInfo,
) -> List[cv2.DMatch]:
  """
  Finds good matches between ORB descriptors of two images using FLANN and ratio test.

  Args:
    img_info1 (ImageInfo): First image's information.
    img_info2 (ImageInfo): Second image's information.

  Returns:
    List[cv2.DMatch]: A list of good matches.
  """
  if img_info1.orb_descriptors is None or img_info2.orb_descriptors is None:
    return []

  matches = flann_matcher.knnMatch(img_info1.orb_descriptors, img_info2.orb_descriptors, k=2)

  # Apply ratio test
  good_matches = []
  for match_pair in matches:
    if len(match_pair) == 2: # Ensure there are two matches for the ratio test
      m, n = match_pair
      if m.distance < 0.7 * n.distance: # Ratio test threshold
        good_matches.append(m)
  return good_matches

def geometric_verification(
  kp1: Tuple[cv2.KeyPoint, ...],
  kp2: Tuple[cv2.KeyPoint, ...],
  matches: List[cv2.DMatch],
) -> Tuple[float, int]:
  """
  Performs geometric verification using RANSAC to compute homography and inlier ratio.

  Args:
    kp1 (Tuple[cv2.KeyPoint, ...]): Keypoints from the first image.
    kp2 (Tuple[cv2.KeyPoint, ...]): Keypoints from the second image.
    matches (List[cv2.DMatch]): List of good matches between keypoints.

  Returns:
    Tuple[float, int]: A tuple containing (inlier_ratio, num_inliers).
  """
  if len(matches) < 4: # RANSAC requires at least 4 points
    return 0.0, 0

  # Extract keypoint coordinates as NumPy arrays
  src_pts = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
  dst_pts = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)

  M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJECTION_THRESHOLD)

  if M is None: # Homography could not be computed
    return 0.0, 0

  num_inliers = int(np.sum(mask)) # Cast to int
  inlier_ratio = float(num_inliers / len(matches)) if len(matches) > 0 else 0.0 # Cast to float

  return inlier_ratio, num_inliers

def orb_geometric_verification(
  img_info1: ImageInfo,
  img_info2: ImageInfo,
) -> bool:
  """
  Performs ORB feature matching and geometric verification to determine similarity.

  Args:
    img_info1 (ImageInfo): First image's information.
    img_info2 (ImageInfo): Second image's information.

  Returns:
    bool: True if images are similar based on ORB+RANSAC, False otherwise.
  """
  # Ensure ORB keypoints and descriptors are loaded
  if img_info1.orb_keypoints is None or img_info1.orb_descriptors is None or \
     img_info2.orb_keypoints is None or img_info2.orb_descriptors is None:
    return False

  good_matches = find_orb_matches(img_info1, img_info2)

  if not good_matches:
    return False

  inlier_ratio, num_inliers = geometric_verification(
    img_info1.orb_keypoints, img_info2.orb_keypoints, good_matches
  )

  if num_inliers >= RANSAC_MIN_INLIERS or inlier_ratio >= RANSAC_INLIER_RATIO_THRESHOLD:
    return True

  return False

def process_images_for_similarity(
  tmp_images_path: str,
) -> List[ImageInfo]:
  """
  Collects all image paths in the temporary directory and pre-processes them.

  Args:
    tmp_images_path (str): Path to the temporary images directory.

  Returns:
    List[ImageInfo]: A list of ImageInfo objects for all images.
  """
  all_image_infos: List[ImageInfo] = []

  image_filenames = [f for f in os.listdir(tmp_images_path) if f.lower().endswith(IMAGE_EXTS)]
  if IMAGE_RANGE_TO_PROCESS is not None:
    image_filenames = image_filenames[IMAGE_RANGE_TO_PROCESS[0]:IMAGE_RANGE_TO_PROCESS[1]]

  for image_filename in tqdm(image_filenames, desc="Pre-processing images"):
    image_path = os.path.join(tmp_images_path, image_filename)
    img_info = load_and_process_image(image_path)
    if img_info:
      all_image_infos.append(img_info)
  return all_image_infos

def find_and_group_similar_images(
  all_image_infos: List[ImageInfo],
  similars_file_path: str,
) -> List[str]:
  """
  Compares all image combinations and groups similar images using DSU.
  Logs similar images to a file and returns a list of images to remove.

  Args:
    all_image_infos (List[ImageInfo]): List of ImageInfo objects.
    similars_file_path (str): Path to the file where similar images will be logged.

  Returns:
    List[str]: A list of image paths that should be removed.
  """
  # Sort images by phash_val to group perceptually similar images together,
  # maximizing cache hits during comparison.
  all_image_infos.sort(key=lambda x: str(x.phash_val))

  image_paths = [info.path for info in all_image_infos]
  dsu = DisjointSetUnion(image_paths)

  num_images = len(all_image_infos)
  total_combinations = num_images * (num_images - 1) // 2
  print(f"Total image combinations to compare: {total_combinations}")

  # Use combinations to avoid duplicate comparisons and ensure each pair is checked once
  for idx1, idx2 in tqdm(cached_combinations(num_images, IMAGE_CACHE_SIZE), desc="Comparing all image combinations...", total=total_combinations):
    img_info1 = all_image_infos[idx1]
    img_info2 = all_image_infos[idx2]

    is_similar, reason = are_images_similar(img_info1, img_info2)
    if is_similar:
      dsu.union(img_info1.path, img_info2.path)
      # Store the reason for logging later
      # For simplicity, we'll associate the reason with the second image in the pair
      # when it's added to a group. This might need refinement if a single image
      # is similar to multiple others for different reasons.
      # For now, we'll just log the reason for the first detected similarity.
      if img_info2.similarity_reason == 'none':
        img_info2.similarity_reason = reason # type: ignore

  similar_images_to_remove: List[str] = []
  with open(similars_file_path, 'w') as f_similars:
    for group in dsu.get_groups():
      if len(group) > 1:
        # Keep the first image in the group, remove the rest
        f_similars.write(f"- {group[0]}\n")
        for path in group[1:]:
          # Find the ImageInfo object for the current path to get its reason
          img_info_to_remove = next((info for info in all_image_infos if info.path == path), None)
          reason_to_log = getattr(img_info_to_remove, 'similarity_reason', 'unknown') if img_info_to_remove else 'unknown'
          f_similars.write(f"  {path}::{reason_to_log}\n")
        similar_images_to_remove.extend(group[1:])
  return similar_images_to_remove

def cached_combinations(num_elements: int, cache_size: int) -> Iterator[Tuple[int, int]]:
  """
  Generates combinations of indices (idx1, idx2) in a cache-friendly order.
  It prioritizes comparisons within cache-sized blocks to maximize cache hits.

  Args:
    num_elements (int): The total number of elements.
    cache_size (int): The size of the cache, used to define block sizes.

  Yields:
    Tuple[int, int]: A pair of indices (idx1, idx2) representing an image combination.
  """
  # Iterate through blocks
  for block_start_idx in range(0, num_elements, cache_size):
    block_end_idx = min(block_start_idx + cache_size, num_elements)

    # Compare elements within the current block
    for i in range(block_start_idx, block_end_idx):
      for j in range(i + 1, block_end_idx):
        yield i, j

    # Compare elements from the current block with elements in subsequent blocks
    for i in range(block_start_idx, block_end_idx):
      for j in range(block_end_idx, num_elements):
        yield i, j

def main():
  """
  Main function to find and remove similar images from a dataset.
  """
  tmp_images_path = os.path.join(TMP_PATH, 'images')
  tmp_labels_path = os.path.join(TMP_PATH, 'label')

  if COPY_IMAGES:
    setup_tmp_directories(TMP_PATH)

    for split_type in SPLIT_NAMES:
      copy_dataset_split(DATASET_PATH, split_type, tmp_images_path, tmp_labels_path)

  all_image_infos = process_images_for_similarity(tmp_images_path)

  similar_images_to_remove = find_and_group_similar_images(
    all_image_infos, SIMILARS_FILE
  )

  print(f"Found {len(similar_images_to_remove)} similar images to remove.")

  # Explicitly close all image handles before attempting to remove files
  for img_info in tqdm(all_image_infos, desc="Closing image file handles..."):
    img_info.close()

  if DELETE_SIMILARS:
    for image_path in tqdm(similar_images_to_remove, desc="Removing similar images and their annotations..."):
      try:
        os.remove(image_path)
        label_filename = os.path.splitext(os.path.basename(image_path))[0] + LABEL_FILE_EXT
        label_path = os.path.join(tmp_labels_path, label_filename)
        if os.path.exists(label_path):
          os.remove(label_path)
      except OSError as e:
        print(f"Error removing {image_path} or its label: {e}")
    print(f"Process complete. Similar images logged in {SIMILARS_FILE} and removed from temporary dataset.")
  else:
    print("DELETE_SIMILARS is False. No images were removed from the dataset.")

if __name__ == '__main__':
  main()
