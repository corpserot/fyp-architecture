import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import sys
import os
import re

def _trim_empty_lines(lines: list[str]) -> list[str]:
  """
  Removes empty strings from the beginning and end of a list of strings.
  """
  start = 0
  end = len(lines) - 1

  while start <= end and not lines[start].strip():
    start += 1
  while end >= start and not lines[end].strip():
    end -= 1

  return lines[start : end + 1]

def _process_cell_content(
    notebook: nbformat.NotebookNode,
    current_cell_type: str | None,
    current_cell_content: list[str],
    markdown_prefix_pattern: re.Pattern,
) -> None:
  """
  Processes and appends the current cell content to the notebook.
  """
  if not current_cell_type or not current_cell_content:
    return

  trimmed_content = _trim_empty_lines(current_cell_content)
  if not trimmed_content:
    return

  if current_cell_type == 'code':
    notebook.cells.append(new_code_cell('\n'.join(trimmed_content)))
  elif current_cell_type == 'markdown':
    processed_markdown = [
      markdown_prefix_pattern.sub('', md_line) if md_line.startswith('#') else md_line
      for md_line in trimmed_content
    ]
    notebook.cells.append(new_markdown_cell('\n'.join(processed_markdown)))

def py_to_ipynb(py_file_path: str) -> None:
  """

  Args:
    py_file_path (str): The path to the input Python (.py) file.
  """
  if not os.path.exists(py_file_path):
    print(f"Error: Input file '{py_file_path}' not found.")
    sys.exit(1)

  py_file_path = os.path.abspath(py_file_path)

  notebook = new_notebook()
  current_cell_type = None
  current_cell_content = []
  in_ipython_block = False # Flag to ignore content within IPython blocks

  # Precompile regex patterns for efficiency
  MAGIC_PATTERN = re.compile(r'# MAGIC: (%[^\s]+.*)')
  BANG_PATTERN = re.compile(r'# BANG: (![^\s]+.*)')
  MARKDOWN_PREFIX_PATTERN = re.compile(r'^#\s*')

  with open(py_file_path, 'r', encoding='utf-8') as f:
    for line in f:
      line = line.rstrip() # Remove trailing whitespace, but keep leading for indentation

      if line == '# BEGIN MARKDOWN':
        _process_cell_content(notebook, current_cell_type, current_cell_content, MARKDOWN_PREFIX_PATTERN)
        current_cell_type = 'markdown'
        current_cell_content = []
      elif line == '# END MARKDOWN':
        _process_cell_content(notebook, current_cell_type, current_cell_content, MARKDOWN_PREFIX_PATTERN)
        current_cell_type = None
        current_cell_content = []
      else:
        if current_cell_type == 'markdown':
          current_cell_content.append(line)
        else:

          if line == '# BEGIN IPYTHON':
            in_ipython_block = True
            continue
          elif line == '# END IPYTHON':
            in_ipython_block = False
            continue

          if in_ipython_block:
            continue # Ignore lines within IPython blocks

          if line == '# NEXT CELL':
            _process_cell_content(notebook, current_cell_type, current_cell_content, MARKDOWN_PREFIX_PATTERN)
            current_cell_type = 'code'
            current_cell_content = []
            continue

          # If not in a markdown block, it's a code cell
          if not current_cell_type: # Start a new code cell if not already in one
            current_cell_type = 'code'
            current_cell_content = []

          # Process special commands using regex
          magic_match = MAGIC_PATTERN.search(line)
          bang_match = BANG_PATTERN.search(line)

          if magic_match:
            current_cell_content.append(magic_match.group(1))
          elif bang_match:
            current_cell_content.append(bang_match.group(1))
          else:
            current_cell_content.append(line)

  # Add any remaining content as a cell
  _process_cell_content(notebook, current_cell_type, current_cell_content, MARKDOWN_PREFIX_PATTERN)

  # Determine output file path
  base_name = os.path.splitext(os.path.basename(py_file_path))[0]
  output_dir = os.path.dirname(py_file_path)
  ipynb_file_path = os.path.join(output_dir, f"{base_name}.ipynb")

  with open(ipynb_file_path, 'w', encoding='utf-8') as f:
    nbformat.write(notebook, f)

  print(f"Successfully converted '{py_file_path}' to '{ipynb_file_path}'")

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("Usage: python py_to_ipynb.py <path>/notebook.py")
    sys.exit(1)

  input_py_file = sys.argv[1]
  py_to_ipynb(input_py_file)
