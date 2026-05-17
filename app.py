"""
Breast Cancer Histopathology Classifier
========================================
Streamlit app for classifying breast histopathology images
as **Benign** or **Malignant** using a Vision Transformer model
with Learned Positional Encoding and Focal Loss.
"""

import streamlit as st
import numpy as np
from PIL import Image

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hero ───────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #1a0a2e 0%, #2d1b69 30%, #6b2fa0 60%, #d63384 100%);
    border-radius: 20px;
    padding: 2.8rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(214,51,132,.3) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,.12);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,.18);
    border-radius: 50px;
    padding: .35rem 1rem;
    font-size: .75rem;
    color: #fff;
    margin-bottom: .8rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-weight: 600;
    position: relative; z-index: 1;
}
.hero h1 {
    font-size: 2.6rem; font-weight: 800; color: #fff;
    margin: 0 0 .4rem; letter-spacing: -.02em;
    position: relative; z-index: 1;
}
.hero p {
    font-size: 1.1rem; color: rgba(255,255,255,.78);
    margin: 0; font-weight: 300; line-height: 1.6;
    position: relative; z-index: 1;
}

/* ── Result cards ───────────────────────────────────── */
.result-card {
    border-radius: 20px; padding: 2rem; margin: 1rem 0;
    overflow: hidden;
}
.result-benign {
    background: linear-gradient(135deg, #0a2e1a, #1b6940, #28a060);
    border: 1px solid rgba(47,160,107,.35);
}
.result-malignant {
    background: linear-gradient(135deg, #2e0a0a, #691b1b, #a03030);
    border: 1px solid rgba(160,47,47,.35);
}
.result-label { font-size: 1.8rem; font-weight: 800; color: #fff; margin: 0 0 .3rem; }
.result-confidence { font-size: 3.2rem; font-weight: 800; color: #fff; margin: 0; line-height: 1; }
.result-conf-label {
    font-size: .8rem; color: rgba(255,255,255,.65);
    text-transform: uppercase; letter-spacing: .1em;
    font-weight: 600; margin-top: .2rem;
}

/* ── Confidence bar ─────────────────────────────────── */
.conf-bar-bg {
    background: rgba(255,255,255,.1); border-radius: 10px;
    height: 12px; margin: 1rem 0; overflow: hidden;
}
.conf-bar { height: 100%; border-radius: 10px; transition: width .8s ease; }
.conf-bar-benign   { background: linear-gradient(90deg, #2fa06b, #4ade80); }
.conf-bar-malignant { background: linear-gradient(90deg, #a02f2f, #ef4444); }

/* ── Upload zone ────────────────────────────────────── */
.upload-zone {
    border: 2px dashed rgba(214,51,132,.35);
    border-radius: 20px; padding: 2.5rem 2rem;
    text-align: center;
    background: linear-gradient(145deg, rgba(214,51,132,.04), rgba(107,47,160,.04));
    transition: all .3s ease;
}
.upload-zone:hover {
    border-color: rgba(214,51,132,.65);
    background: linear-gradient(145deg, rgba(214,51,132,.08), rgba(107,47,160,.08));
}

/* ── Info / sidebar cards ───────────────────────────── */
.info-card {
    background: linear-gradient(145deg, rgba(255,255,255,.05), rgba(255,255,255,.02));
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 16px; padding: 1.4rem; margin: .5rem 0;
    backdrop-filter: blur(10px);
}
.info-card h4 { color: #d63384; margin: 0 0 .5rem; font-weight: 600; }
.info-card p  { color: rgba(255,255,255,.68); margin: 0; font-size: .88rem; line-height: 1.55; }

.sidebar-section {
    background: linear-gradient(145deg, rgba(214,51,132,.07), rgba(107,47,160,.07));
    border: 1px solid rgba(214,51,132,.13);
    border-radius: 12px; padding: 1.2rem; margin: 1rem 0;
}
.sidebar-section h4 {
    color: #d63384; margin: 0 0 .7rem; font-size: .85rem;
    text-transform: uppercase; letter-spacing: .06em; font-weight: 700;
}
.metric-row {
    display: flex; justify-content: space-between;
    padding: .35rem 0; border-bottom: 1px solid rgba(255,255,255,.05);
    font-size: .83rem;
}
.metric-row:last-child { border-bottom: none; }
.metric-label { color: rgba(255,255,255,.5); }
.metric-value { color: #fff; font-weight: 600; }

/* ── Disclaimer ─────────────────────────────────────── */
.disclaimer {
    background: rgba(255,193,7,.07);
    border: 1px solid rgba(255,193,7,.18);
    border-radius: 12px; padding: 1rem 1.2rem;
    margin: 1.5rem 0; font-size: .84rem;
    color: rgba(255,255,255,.68); line-height: 1.55;
}
.disclaimer strong { color: #ffc107; }

.footer {
    text-align: center; padding: 2rem 0 1rem;
    color: rgba(255,255,255,.25); font-size: .78rem;
    border-top: 1px solid rgba(255,255,255,.05); margin-top: 3rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Classifier Settings")

    st.markdown(
        """
    <div class="sidebar-section">
        <h4>Model Architecture</h4>
        <div class="metric-row">
            <span class="metric-label">Type</span>
            <span class="metric-value">Vision Transformer</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Positional Enc.</span>
            <span class="metric-value">Learned</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Loss Function</span>
            <span class="metric-value">Focal Loss</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Input Size</span>
            <span class="metric-value">224 × 224</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Classes</span>
            <span class="metric-value">Benign · Malignant</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.50,
        max_value=0.99,
        value=0.70,
        step=0.01,
        help="Predictions below this threshold will be flagged as uncertain.",
    )

    st.markdown("---")
    st.markdown(
        """
    <div class="sidebar-section">
        <h4>Quick Start</h4>
        <div style="color:rgba(255,255,255,.65); font-size:.83rem; line-height:1.6;">
            <b>1.</b> Set <code>MODEL_PATH</code>, <code>MODEL_GDRIVE_ID</code>,
            or <code>MODEL_URL</code> env var.<br>
            <b>2.</b> Upload a histopathology image.<br>
            <b>3.</b> Click <b>Classify</b> to get the prediction.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="disclaimer">
        <strong>⚠ Medical Disclaimer</strong><br>
        This tool is for <u>research &amp; educational purposes only</u>.
        It is <b>not</b> a substitute for professional medical diagnosis.
        Always consult a qualified pathologist.
    </div>
    """,
        unsafe_allow_html=True,
    )


# ── Hero ─────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
    <div class="hero-badge">AI-Powered Pathology</div>
    <h1>Breast Cancer Classifier</h1>
    <p>Upload a breast histopathology image and the Vision Transformer model
    will classify it as <b>Benign</b> or <b>Malignant</b> with a confidence score.</p>
</div>
""",
    unsafe_allow_html=True,
)


# ── Model loading ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_model():
    from model_loader import load_model

    return load_model()


model = None
model_error = None

try:
    with st.spinner("🧠 Loading model — this may take a moment on first run …"):
        model = _load_model()
except FileNotFoundError as exc:
    model_error = str(exc)
except Exception as exc:  # noqa: BLE001
    model_error = f"Failed to load model: {exc}"

if model_error:
    st.warning("⚙️ **Model not loaded.** " + model_error)


# ── Upload & classify ───────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("### 📤 Upload Image")
    st.markdown(
        '<div class="upload-zone">'
        "<p style='font-size:1.05rem;color:rgba(255,255,255,.55);margin:0;'>"
        "Drag &amp; drop or browse for a histopathology image<br>"
        "<span style='font-size:.82rem;'>Supported: PNG · JPG · JPEG · BMP · TIFF</span>"
        "</p></div>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)


with col_result:
    st.markdown("### 🧬 Classification Result")

    if uploaded_file is not None and model is not None:
        classify_btn = st.button(
            "🔍  Classify Image", use_container_width=True, type="primary"
        )

        if classify_btn:
            from model_loader import preprocess_image, CLASS_NAMES, IMG_SIZE

            with st.spinner("Analysing …"):
                processed = preprocess_image(image)
                prediction = model.predict(processed, verbose=0)

            # Handle both sigmoid (shape [1,1]) and softmax (shape [1,2]) outputs
            if prediction.shape[-1] == 1:
                malignant_prob = float(prediction[0][0])
                benign_prob = 1.0 - malignant_prob
            else:
                benign_prob = float(prediction[0][0])
                malignant_prob = float(prediction[0][1])

            predicted_idx = int(malignant_prob >= 0.5)
            label = CLASS_NAMES[predicted_idx]
            confidence = malignant_prob if predicted_idx == 1 else benign_prob

            # Card colour
            card_cls = "result-malignant" if predicted_idx == 1 else "result-benign"
            bar_cls = "conf-bar-malignant" if predicted_idx == 1 else "conf-bar-benign"
            emoji = "⚠️" if predicted_idx == 1 else "✅"

            st.markdown(
                f"""
            <div class="result-card {card_cls}">
                <div class="result-label">{emoji}  {label}</div>
                <div class="result-confidence">{confidence * 100:.1f}%</div>
                <div class="result-conf-label">Confidence</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar {bar_cls}" style="width:{confidence * 100:.1f}%"></div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Uncertainty flag
            if confidence < confidence_threshold:
                st.warning(
                    f"⚡ Confidence ({confidence:.1%}) is below threshold "
                    f"({confidence_threshold:.0%}). Consider manual review."
                )

            # Detailed probabilities
            with st.expander("📊 Detailed probabilities"):
                prob_col1, prob_col2 = st.columns(2)
                with prob_col1:
                    st.metric("Benign", f"{benign_prob:.2%}")
                with prob_col2:
                    st.metric("Malignant", f"{malignant_prob:.2%}")
                st.caption(f"Input resized to {IMG_SIZE}×{IMG_SIZE} · Pixel values normalised to [0, 1]")

    elif uploaded_file is not None and model is None:
        st.info("⚙️ Load a model first to classify images.")
    else:
        st.markdown(
            """
        <div class="info-card">
            <h4>No image uploaded</h4>
            <p>Upload a breast histopathology slide image in the panel on the
            left, then click <b>Classify</b> to run inference.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ── Information cards ────────────────────────────────────────────────
st.markdown("---")
st.markdown("### ℹ️  About This Tool")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
    <div class="info-card">
        <h4>🔬 Vision Transformer</h4>
        <p>Uses a ViT backbone with <b>Learned Positional Encoding</b> to
        capture spatial relationships between tissue patches, achieving
        state-of-the-art accuracy on histopathology data.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
    <div class="info-card">
        <h4>⚖️ Focal Loss</h4>
        <p>Trained with <b>Focal Loss</b> (γ = 2, α = 0.25) to mitigate
        class imbalance — the model focuses learning on hard-to-classify
        samples rather than easy negatives.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
    <div class="info-card">
        <h4>🛡️ Responsible AI</h4>
        <p>Predictions include confidence scores and an adjustable
        uncertainty threshold. Low-confidence results are flagged
        for manual pathologist review.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ── Footer ───────────────────────────────────────────────────────────
st.markdown(
    """
<div class="footer">
    Breast Cancer Classifier · Built with Streamlit &amp; TensorFlow<br>
    For research &amp; educational purposes only.
</div>
""",
    unsafe_allow_html=True,
)
