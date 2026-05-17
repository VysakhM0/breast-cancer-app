import numpy as np
import streamlit as st
from PIL import Image

from model_loader import load_model

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BreastPath AI",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Dark background */
    .stApp {
        background: #080c14;
    }

    /* Top banner */
    .banner {
        background: linear-gradient(135deg, #0d1f3c 0%, #0a1628 60%, #061020 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 2.5rem 2rem 2rem 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .banner h1 {
        color: #e8f4ff;
        font-size: 2.2rem;
        font-weight: 600;
        letter-spacing: -0.5px;
        margin: 0 0 0.4rem 0;
    }
    .banner p {
        color: #7aa8d4;
        font-size: 0.95rem;
        margin: 0;
    }
    .badge {
        display: inline-block;
        background: #0f2744;
        border: 1px solid #1e5f8c;
        border-radius: 20px;
        color: #4db8ff;
        font-size: 0.75rem;
        font-family: 'DM Mono', monospace;
        padding: 0.2rem 0.75rem;
        margin-bottom: 1rem;
        letter-spacing: 0.5px;
    }

    /* Upload box */
    .upload-area {
        background: #0d1829;
        border: 2px dashed #1e3a5f;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
        transition: border-color 0.2s;
    }

    /* Result cards */
    .result-card {
        background: #0d1829;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #1e3a5f;
    }
    .result-benign {
        border-left: 4px solid #00e5a0;
    }
    .result-malignant {
        border-left: 4px solid #ff4d6d;
    }
    .result-label {
        font-size: 1.6rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .label-benign  { color: #00e5a0; }
    .label-malignant { color: #ff4d6d; }
    .confidence-text {
        color: #7aa8d4;
        font-size: 0.9rem;
    }

    /* Probability bar */
    .prob-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.6rem;
    }
    .prob-label {
        color: #c8dff5;
        font-size: 0.85rem;
        width: 90px;
        flex-shrink: 0;
    }
    .prob-bar-bg {
        flex: 1;
        background: #0a1628;
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
    }
    .prob-bar-fill-b {
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, #00b87a, #00e5a0);
        transition: width 0.6s ease;
    }
    .prob-bar-fill-m {
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, #cc1f3f, #ff4d6d);
        transition: width 0.6s ease;
    }
    .prob-pct {
        color: #7aa8d4;
        font-family: 'DM Mono', monospace;
        font-size: 0.82rem;
        width: 44px;
        text-align: right;
        flex-shrink: 0;
    }

    /* Disclaimer */
    .disclaimer {
        background: #0d1829;
        border: 1px solid #2a1f00;
        border-left: 4px solid #f0a500;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        color: #c8a84b;
        font-size: 0.82rem;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #2a4a6e;
        font-size: 0.78rem;
        padding: 2rem 0 1rem 0;
        font-family: 'DM Mono', monospace;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
    header    {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Banner ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="banner">
        <div class="badge">EfficientNetB3 + Attention + Bi-LSTM</div>
        <h1>🔬 BreastPath AI</h1>
        <p>Histopathology image classification · Benign vs Malignant · BreaKHis Dataset</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="disclaimer">
        ⚠️ <strong>Research use only.</strong>
        This tool is not a certified medical device and must not be used as a substitute
        for professional clinical diagnosis. Always consult a qualified pathologist.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load model ────────────────────────────────────────────────────────────────
model = load_model()

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload an H&E-stained histopathology image",
    type=["jpg", "jpeg", "png"],
    help="Accepts JPG or PNG. The image will be resized to 224×224 internally.",
)

if uploaded is None:
    st.markdown(
        """
        <div style="text-align:center; padding: 2.5rem 0; color:#2a4a6e;">
            <div style="font-size:3rem; margin-bottom:0.5rem;">🖼️</div>
            <div style="font-size:0.9rem;">Upload an image above to get started</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # ── Show uploaded image ───────────────────────────────────────────────────
    image = Image.open(uploaded).convert("RGB")
    col_img, col_gap = st.columns([2, 1])
    with col_img:
        st.image(image, caption="Uploaded histopathology image", use_column_width=True)

    # ── Preprocess ────────────────────────────────────────────────────────────
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

    # ── Predict ───────────────────────────────────────────────────────────────
    with st.spinner("Analysing image..."):
        prob_malignant = float(model.predict(img_array, verbose=0)[0][0])

    prob_benign = 1.0 - prob_malignant
    label       = "Malignant" if prob_malignant >= 0.5 else "Benign"
    confidence  = prob_malignant if label == "Malignant" else prob_benign

    # ── Result card ───────────────────────────────────────────────────────────
    card_cls  = "result-malignant" if label == "Malignant" else "result-benign"
    label_cls = "label-malignant"  if label == "Malignant" else "label-benign"
    icon      = "🔴" if label == "Malignant" else "🟢"

    st.markdown(
        f"""
        <div class="result-card {card_cls}">
            <div class="result-label {label_cls}">{icon} {label}</div>
            <div class="confidence-text">Confidence: <strong>{confidence:.1%}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Probability bars ──────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="result-card" style="padding-top:1.2rem;">
            <div style="color:#7aa8d4; font-size:0.8rem; margin-bottom:0.9rem;
                        text-transform:uppercase; letter-spacing:1px;">
                Class Probabilities
            </div>
            <div class="prob-row">
                <span class="prob-label">Benign</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill-b" style="width:{prob_benign*100:.1f}%"></div>
                </div>
                <span class="prob-pct">{prob_benign:.1%}</span>
            </div>
            <div class="prob-row">
                <span class="prob-label">Malignant</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill-m" style="width:{prob_malignant*100:.1f}%"></div>
                </div>
                <span class="prob-pct">{prob_malignant:.1%}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Raw score (for researchers) ───────────────────────────────────────────
    with st.expander("Show raw model output"):
        st.code(
            f"Raw sigmoid output : {prob_malignant:.6f}\n"
            f"Decision threshold : 0.5000\n"
            f"Final prediction   : {label}",
            language="text",
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer">
        BreastPath AI · EfficientNetB3 + Bi-LSTM · BreaKHis Dataset ·
        Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)