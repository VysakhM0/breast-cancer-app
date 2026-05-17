# Breast Cancer Classifier

A deep learning-powered breast cancer classification application built with Streamlit and TensorFlow/Keras.

## Features

- **Image Classification**: Upload histopathology images for breast cancer classification
- **Vision Transformer Architecture**: Uses custom Keras layers including Learned Positional Encoding
- **Focal Loss**: Handles class imbalance in medical imaging datasets
- **Interactive UI**: Streamlit-based web interface for easy interaction

## Project Structure

```
├── app.py              # Streamlit application
├── model_loader.py     # Model loading and inference utilities
├── custom_layers.py    # Custom Keras layers (LearnedPositionalEncoding, focal_loss)
├── requirements.txt    # Python dependencies
└── .streamlit/         # Streamlit configuration
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   streamlit run app.py
   ```

## Requirements

- Python 3.8+
- TensorFlow/Keras
- Streamlit
- See `requirements.txt` for full list
