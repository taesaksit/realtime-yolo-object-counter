import os

from ultralytics import YOLO

MODEL_FOLDER = "./models"
DEFAULT_MODEL = "model-book-best.pt"


def get_available_models():
    """
    Return all .pt models inside models/
    """
    if not os.path.exists(MODEL_FOLDER):
        return []

    models = [
        filename
        for filename in os.listdir(MODEL_FOLDER)
        if filename.lower().endswith(".pt")
    ]

    return sorted(models)


def get_default_model():
    models = get_available_models()

    if DEFAULT_MODEL in models:
        return DEFAULT_MODEL

    if models:
        return models[0]

    return ""


def load_model(model_name):
    """
    Load selected YOLO model.

    Only models existing inside MODEL_FOLDER
    are allowed.
    """
    available_models = get_available_models()

    if model_name not in available_models:
        return None

    model_path = os.path.join(MODEL_FOLDER, model_name)

    print(f"[INFO] Loading model: {model_path}")

    return YOLO(model_path)
