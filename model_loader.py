"""
Model download, caching, and loading utilities.

Handles:
    - Downloading model weights from Google Drive or direct URLs
    - Local caching to avoid repeated downloads
    - Loading Keras models with custom objects (LearnedPositionalEncoding, focal_loss)
"""

import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from custom_layers import LearnedPositionalEncoding, focal_loss


# ── Configuration ────────────────────────────────────────────────────
# Set your model source via environment variables or edit defaults here.
#   MODEL_PATH       → absolute path to a local .h5 / SavedModel
#   MODEL_GDRIVE_ID  → Google Drive file ID (uses gdown)
#   MODEL_URL        → direct-download URL
GDRIVE_FILE_ID = os.environ.get("MODEL_GDRIVE_ID", "")
MODEL_URL = os.environ.get("MODEL_URL", "")
LOCAL_MODEL_PATH = os.environ.get("MODEL_PATH", "")

CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "breast_cancer_classifier"
)
CACHED_MODEL_FILE = os.path.join(CACHE_DIR, "breast_cancer_model.h5")

# Must match training configuration
IMG_SIZE = int(os.environ.get("IMG_SIZE", "224"))
CLASS_NAMES = ["Benign", "Malignant"]


# ── Helpers ──────────────────────────────────────────────────────────
def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _download_from_gdrive(file_id: str, dest: str) -> str:
    import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest, quiet=False)
    return dest


def _download_from_url(url: str, dest: str) -> str:
    import urllib.request

    urllib.request.urlretrieve(url, dest)
    return dest


# ── Public API ───────────────────────────────────────────────────────
def get_model_path() -> str:
    """
    Resolve the model file path, downloading if necessary.

    Priority:
        1. Local file path  (MODEL_PATH env var)
        2. Previously cached download
        3. Google Drive      (MODEL_GDRIVE_ID env var)
        4. Direct URL        (MODEL_URL env var)

    Returns:
        str: Absolute path to the model file.

    Raises:
        FileNotFoundError: If no model source is configured.
    """
    if LOCAL_MODEL_PATH and os.path.isfile(LOCAL_MODEL_PATH):
        return LOCAL_MODEL_PATH

    if os.path.isfile(CACHED_MODEL_FILE):
        return CACHED_MODEL_FILE

    _ensure_cache_dir()

    if GDRIVE_FILE_ID:
        st.info("📥 Downloading model from Google Drive …")
        return _download_from_gdrive(GDRIVE_FILE_ID, CACHED_MODEL_FILE)

    if MODEL_URL:
        st.info("📥 Downloading model …")
        return _download_from_url(MODEL_URL, CACHED_MODEL_FILE)

    raise FileNotFoundError(
        "No model found. Set one of these environment variables:\n"
        "  MODEL_PATH       → path to a local .h5 / SavedModel\n"
        "  MODEL_GDRIVE_ID  → Google Drive file ID\n"
        "  MODEL_URL        → direct download URL"
    )


@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load the Keras model with custom objects.
    Result is cached across Streamlit reruns.
    """
    model_path = get_model_path()
    custom_objects = {
        "LearnedPositionalEncoding": LearnedPositionalEncoding,
        "focal_loss_fn": focal_loss(gamma=2.0, alpha=0.25),
    }
    model = tf.keras.models.load_model(
        model_path, custom_objects=custom_objects
    )
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess a PIL image for model inference.

    Args:
        image: Input PIL Image (any mode / size).

    Returns:
        np.ndarray of shape (1, IMG_SIZE, IMG_SIZE, 3), float32 in [0, 1].
    """
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)
