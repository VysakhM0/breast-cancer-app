import os

import gdown
import streamlit as st
import tensorflow as tf

from custom_layers import LearnedPositionalEncoding, focal_loss

# ── PASTE YOUR GOOGLE DRIVE FILE ID HERE ─────────────────────────────────────
# Example: if your share link is
#   https://drive.google.com/file/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlso/view
# then your FILE_ID is:
#   1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlso
GDRIVE_FILE_ID = "1kzVzZFS07s84c2pT8OT4Z9zymSYdvxgY"

MODEL_PATH = "final_hybrid_model.keras"


@st.cache_resource(show_spinner="Loading model weights — this takes ~30 s on first run...")
def load_model():
    """Download model from Google Drive (once) then load it."""

    if not os.path.exists(MODEL_PATH):
        if GDRIVE_FILE_ID == "YOUR_FILE_ID_HERE":
            st.error(
                "⚠️  Model not configured. "
                "Open model_loader.py and paste your Google Drive file ID into GDRIVE_FILE_ID."
            )
            st.stop()

        with st.spinner("Downloading model from Google Drive..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "LearnedPositionalEncoding": LearnedPositionalEncoding,
            "focal_loss": focal_loss(gamma=1.5, alpha=0.5),
            "loss_fn": focal_loss(gamma=1.5, alpha=0.5),
        },
    )
    return model