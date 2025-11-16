# python ./generate_palette.py

import math
import os
from extracolormath import (
  adjust_chroma_to_fit_gamut,
)

SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
PALETTE_FILE = os.path.join(SCRIPTDIR, "palette.csv")

COUNT_HUE_STEPS = 8
COUNT_LIGHTNESS_STEPS = 5
COUNT_CHROMA_STEPS = 3
LIGHTNESS_MIN=0.4
LIGHTNESS_MAX=0.95
CHROMA_MIN = 0.05
CHROMA_MAX = 0.25

def generate_cylindrical_palette(
  count_hue_steps: int,
  count_lightness_steps: int,
  count_chroma_steps: int,
  L_min: float,
  L_max: float,
  C_min: float,
  C_max: float,
) -> list[list[list[tuple[float, float, float]]]]:
  """
  Produce a three dimensional palette arranged in a cylindrical pattern.
  Horizontal axis varies hue uniformly.
  Vertical axis varies lightness uniformly.
  Depth axis varies chroma uniformly.
  Each cell is an Oklch tuple (L, C, h).

  Args:
    count_hue_steps: Number of steps for hue variation.
    count_lightness_steps: Number of steps for lightness variation.
    count_chroma_steps: Number of steps for chroma variation.
    L_min: Minimum lightness value (0-1).
    L_max: Maximum lightness value (0-1).
    C_min: Minimum chroma value.
    C_max: Maximum chroma value.

  Returns:
    A list of lists of lists, where the outer list represents chroma slices,
    the middle list represents lightness rows, and the inner list represents hue columns.
    Each innermost element is an Oklch color tuple (L, C, h).
  """

  palette = []

  # build uniform grids
  for k in range(count_chroma_steps):
    chroma_slice = []
    t_C = k / (count_chroma_steps - 1) if count_chroma_steps > 1 else 0.0
    C_target = C_min + t_C * (C_max - C_min)

    for i in range(count_lightness_steps):
      row = []
      t_L = i / (count_lightness_steps - 1) if count_lightness_steps > 1 else 0.0
      L = L_min + t_L * (L_max - L_min)

      for j in range(count_hue_steps):
        h = (j / count_hue_steps) * 360.0

        # Adjust chroma to fit within the sRGB gamut for the given L and h
        C = adjust_chroma_to_fit_gamut(L, C_target, h)

        # The output is now Oklch (L, C, h)
        row.append((L, C, h))
      chroma_slice.append(row)
    palette.append(chroma_slice)

  return palette


def write_palette_to_csv(
  palette: list[list[list[tuple[float, float, float]]]],
  filepath: str,
) -> None:
  """
  Writes a 3D color palette to a CSV file.

  Each row in the CSV represents a lightness-hue grid for a specific chroma slice.
  The columns are (L, C, h) values.

  Args:
    palette: The 3D list of Oklch color tuples.
    filepath: The path to the output CSV file.
  """
  import csv

  with open(filepath, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for chroma_slice in palette:
      for row in chroma_slice:
        # Flatten each color tuple into a list of floats for the CSV row
        flat_row = [val for color_tuple in row for val in color_tuple]
        writer.writerow(flat_row)


def read_palette_from_csv(
  filepath: str,
  count_lightness_steps: int, # New parameter to reconstruct 3D structure
  count_hue_steps: int, # New parameter to reconstruct 3D structure
) -> list[list[list[tuple[float, float, float]]]]:
  """
  Reads a 3D color palette from a CSV file.

  Args:
    filepath: The path to the input CSV file.
    count_lightness_steps: Number of lightness steps used to generate the palette.
    count_hue_steps: Number of hue steps used to generate the palette.

  Returns:
    A 3D list of Oklch color tuples (L, C, h).
  """
  import csv

  palette = []
  current_chroma_slice = []
  current_row_in_slice = 0

  with open(filepath, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row_data in reader:
      color_row = []
      for i in range(0, len(row_data), 3):
        L = float(row_data[i])
        C = float(row_data[i+1])
        h = float(row_data[i+2])
        color_row.append((L, C, h))

      current_chroma_slice.append(color_row)
      current_row_in_slice += 1

      if current_row_in_slice == count_lightness_steps:
        palette.append(current_chroma_slice)
        current_chroma_slice = []
        current_row_in_slice = 0
  return palette


def main():
  palette = generate_cylindrical_palette(
      count_hue_steps=COUNT_HUE_STEPS,
      count_lightness_steps=COUNT_LIGHTNESS_STEPS,
      count_chroma_steps=COUNT_CHROMA_STEPS,
      L_min=LIGHTNESS_MIN,
      L_max=LIGHTNESS_MAX,
      C_min=CHROMA_MIN,
      C_max=CHROMA_MAX,
  )

  # Add a greyscale gradient row
  greyscale_row = []
  greyscale_min = max(LIGHTNESS_MIN-0.1, 0)
  greyscale_max = min(LIGHTNESS_MAX+0.1, 1)
  for i in range(COUNT_HUE_STEPS):
    L_grey = greyscale_min + (i / (COUNT_HUE_STEPS - 1)) * (greyscale_max - greyscale_min)
    greyscale_row.append((L_grey, 0.0, 0.0))
  palette.append([greyscale_row]) # Append as a new "chroma slice" with one row

  write_palette_to_csv(palette, PALETTE_FILE)

if __name__ == "__main__":
  main()
