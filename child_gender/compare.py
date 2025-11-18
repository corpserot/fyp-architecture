import os
from ultralytics import YOLO

SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))

# Define constants for model paths and dataset path
# Assuming models are in child_gender/models/ and dataset is in child_gender/dataset/
# Adjust these paths based on your actual project structure
MODELS = {
    "MODEL_1": os.path.join(SCRIPTDIR, "models/model-1.pt"),
    "MODEL_2": os.path.join(SCRIPTDIR, "models/model-2.pt"),
    # Add more models as needed
}
DATASET_PATH = os.path.join(SCRIPTDIR, "dataset_bbox2_yolo2_processed/data.yaml") # Path to your dataset's data.yaml for the test split

def compare_models(models: dict, dataset_path: str) -> None:
    """
    Loads YOLO models and runs them against a test split dataset.
    Prints relevant metrics using the .val() method.

    Args:
        models (dict): A dictionary where keys are model names and values are their paths.
        dataset_path (str): The path to the dataset's data.yaml file for validation.
    """
    for name, path in models.items():
        print(f"--- Evaluating {name} ---")
        try:
            model = YOLO(path)
            model.val(data=dataset_path)
            print("-" * 30)
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            print("-" * 30)

if __name__ == "__main__":
    compare_models(MODELS, DATASET_PATH)
