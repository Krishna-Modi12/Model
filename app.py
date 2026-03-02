"""
app.py — Premium Face Shape AI · Gradio Web Demo
==================================================
Built from Stitch-designed UI screens with glassmorphism dark theme.

Usage:
    python app.py                 # http://localhost:7860
    python app.py --share         # Public Gradio link
    python app.py --port 8080     # Custom port

Discovery (from predict.py):
    Inference function : predict_single(image_path, model, extractor, device)
    Returns bbox       : YES — pixel coords (x, y, w, h)
    Device handling    : predict.py manages GPU/CPU internally
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path

import torch
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ═══════════════════════════════════════════════════════════════
# CONFIG — All constants, paths, class info in one place
# ═══════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

APP_TITLE       = "Face Shape AI"
APP_SUBTITLE    = "Upload a photo · Get your face shape instantly"
MODEL_ACCURACY  = "79.21%"
MODEL_NAME      = "celeba_v5_epoch=14_val_f1=0.7654"
DEMO_DIR        = PROJECT_ROOT / "demo_examples"

FACE_SHAPE_INFO = {
    "Heart":  {"icon": "💜", "color": "#E879F9", "desc": "Wider forehead, narrow pointed chin"},
    "Oblong": {"icon": "📏", "color": "#A78BFA", "desc": "Long and narrow with balanced width"},
    "Oval":   {"icon": "🥚", "color": "#67E8F9", "desc": "Slightly wider cheekbones, balanced"},
    "Round":  {"icon": "⭕", "color": "#34D399", "desc": "Equal width and length, soft angles"},
    "Square": {"icon": "⬜", "color": "#F59E0B", "desc": "Strong jawline, equal proportions"},
}

# Design DNA (extracted from Stitch screens)
COLORS = {
    "bg":       "#0F1117",
    "card":     "rgba(26, 31, 46, 0.7)",
    "card_solid": "#1A1F2E",
    "accent1":  "#7C3AED",  # purple
    "accent2":  "#2BCDEE",  # cyan
    "success":  "#10B981",
    "error":    "#EF4444",
    "text":     "#F1F5F9",
    "text_dim": "#94A3B8",
    "border":   "rgba(124, 58, 237, 0.3)",
}


# ═══════════════════════════════════════════════════════════════
# CSS — Premium dark theme inspired by Stitch glassmorphism
# ═══════════════════════════════════════════════════════════════

CUSTOM_CSS = """
/* ── Global ────────────────────────────────── */
.gradio-container {
    background: #0F1117 !important;
    font-family: 'Inter', sans-serif !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}
.dark { background: #0F1117 !important; }

/* ── Header ────────────────────────────────── */
.app-header {
    text-align: center;
    padding: 2rem 1rem 1rem;
}
.app-header h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #7C3AED, #2BCDEE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
    letter-spacing: -0.02em;
}
.app-header .subtitle {
    color: #94A3B8;
    font-size: 1.05rem;
    font-weight: 400;
}
.app-header .badge {
    display: inline-block;
    background: rgba(124, 58, 237, 0.15);
    border: 1px solid rgba(124, 58, 237, 0.3);
    color: #A78BFA;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 0.5rem;
}

/* ── Glass Cards ───────────────────────────── */
.glass-card {
    background: rgba(26, 31, 46, 0.65) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(124, 58, 237, 0.2) !important;
    border-radius: 16px !important;
    padding: 1.25rem !important;
    transition: all 0.3s ease !important;
}
.glass-card:hover {
    border-color: rgba(43, 205, 238, 0.4) !important;
    box-shadow: 0 0 20px rgba(43, 205, 238, 0.08) !important;
}

/* ── Shape Info Cards ──────────────────────── */
.shape-cards {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    justify-content: center;
    margin: 1.5rem 0;
}
.shape-card {
    background: rgba(26, 31, 46, 0.5);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(124, 58, 237, 0.15);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    min-width: 180px;
    flex: 1;
    text-align: center;
    transition: all 0.3s ease;
}
.shape-card:hover {
    border-color: rgba(43, 205, 238, 0.5);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.shape-card .icon { font-size: 1.6rem; margin-bottom: 0.3rem; }
.shape-card .name {
    font-weight: 700;
    color: #F1F5F9;
    font-size: 0.95rem;
    margin-bottom: 0.2rem;
}
.shape-card .desc {
    color: #94A3B8;
    font-size: 0.78rem;
    line-height: 1.4;
}

/* ── Prediction Badge ──────────────────────── */
.prediction-badge {
    text-align: center;
    padding: 1rem;
}
.prediction-badge .shape-name {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #7C3AED, #2BCDEE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.prediction-badge .confidence {
    color: #10B981;
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 0.25rem;
}

/* ── Footer ────────────────────────────────── */
.app-footer {
    text-align: center;
    padding: 1.5rem 0;
    color: #64748B;
    font-size: 0.82rem;
    border-top: 1px solid rgba(124, 58, 237, 0.1);
    margin-top: 1.5rem;
}
.app-footer strong { color: #94A3B8; }

/* ── Gradio Overrides ──────────────────────── */
.gr-button.primary {
    background: linear-gradient(135deg, #7C3AED, #2BCDEE) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
}
.gr-button.primary:hover {
    opacity: 0.9 !important;
    box-shadow: 0 4px 16px rgba(124, 58, 237, 0.4) !important;
}
.gr-input, .gr-box, .gr-panel {
    background: rgba(26, 31, 46, 0.5) !important;
    border-color: rgba(124, 58, 237, 0.2) !important;
    border-radius: 12px !important;
}
.label-wrap {
    color: #94A3B8 !important;
}

/* ── Error message ─────────────────────────── */
.error-msg {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 12px;
    padding: 1rem;
    color: #FCA5A5;
}
.error-msg h3 { color: #EF4444; margin-bottom: 0.5rem; }
.error-msg ul { padding-left: 1.2rem; color: #94A3B8; }
"""


# ═══════════════════════════════════════════════════════════════
# MODEL — Load model and extractor once at startup
# ═══════════════════════════════════════════════════════════════

from predict import load_model, predict_single, DEFAULT_CHECKPOINT
from src.utils.landmark_extractor import LandmarkExtractor

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL     = None
EXTRACTOR = None

def init_model():
    """Load model and extractor once at startup."""
    global MODEL, EXTRACTOR
    print(f"  ✦ Loading model on: {DEVICE}")
    MODEL = load_model(DEFAULT_CHECKPOINT, DEVICE)
    EXTRACTOR = LandmarkExtractor(static_image_mode=True)
    print(f"  ✦ Model ready: {MODEL_NAME}")


# ═══════════════════════════════════════════════════════════════
# INFERENCE — Wrapper around predict.py (NO duplicate logic)
# ═══════════════════════════════════════════════════════════════

def run_inference(pil_image: Image.Image):
    """
    Accepts PIL Image from Gradio → temp file → predict_single() →
    returns (annotated_image, shape_label_html, confidence_scores, json_result).
    """
    if pil_image is None:
        return (None, "", {}, {"status": "awaiting_upload"})

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            pil_image.save(tmp, format="JPEG", quality=95)
            tmp_path = tmp.name

        results = predict_single(tmp_path, MODEL, EXTRACTOR, DEVICE)

    except Exception as e:
        return (
            pil_image,
            _error_html("Something went wrong", str(e)),
            {},
            {"error": str(e)},
        )
    finally:
        if tmp_path:
            try: os.unlink(tmp_path)
            except: pass

    # Handle errors
    if not results or "error" in results[0]:
        error_msg = results[0].get("error", "Unknown error") if results else "No results"
        return (
            pil_image,
            _error_html("No Face Detected", error_msg),
            {},
            {"error": error_msg},
        )

    # ── Success ──
    r = results[0]
    annotated = draw_bounding_box(pil_image, r)
    label_html = _prediction_html(r["predicted_class"], r["confidence"])
    scores = {k: float(v) for k, v in r["all_scores"].items()}

    json_out = dict(r)
    if "bbox" in json_out:
        json_out["bbox"] = list(json_out["bbox"])

    return (annotated, label_html, scores, json_out)


def _prediction_html(shape: str, conf: float) -> str:
    info = FACE_SHAPE_INFO.get(shape, {})
    icon = info.get("icon", "✨")
    desc = info.get("desc", "")
    return f"""
    <div class="prediction-badge">
        <div style="font-size: 2.5rem; margin-bottom: 0.2rem;">{icon}</div>
        <div class="shape-name">{shape}</div>
        <div class="confidence">✓ {conf * 100:.1f}% confidence</div>
        <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 0.4rem;">{desc}</div>
    </div>
    """

def _error_html(title: str, detail: str) -> str:
    return f"""
    <div class="error-msg">
        <h3>⚠ {title}</h3>
        <p>{detail}</p>
        <ul>
            <li>Use a clear, front-facing photo</li>
            <li>Ensure good lighting</li>
            <li>Remove sunglasses or hats</li>
            <li>Slight tilts are OK</li>
        </ul>
    </div>
    """


# ═══════════════════════════════════════════════════════════════
# VISUALIZATION — Bounding box drawing
# ═══════════════════════════════════════════════════════════════

def draw_bounding_box(pil_image: Image.Image, result: dict) -> Image.Image:
    """Draw bbox + label on image. bbox is pixel coords (x, y, w, h)."""
    bbox = result.get("bbox")
    if bbox is None:
        return pil_image

    img = pil_image.copy()
    draw = ImageDraw.Draw(img)

    x, y, w, h = bbox
    x1, y1, x2, y2 = x, y, x + w, y + h

    # Accent green box with 3px width
    box_color = "#10B981"
    draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

    # Label
    label = f"{result['predicted_class']} ({result['confidence'] * 100:.1f}%)"
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except (OSError, IOError):
        font = ImageFont.load_default()

    text_bb = draw.textbbox((x1, y1 - 22), label, font=font)
    draw.rectangle(
        [text_bb[0] - 3, text_bb[1] - 2, text_bb[2] + 5, text_bb[3] + 2],
        fill=box_color,
    )
    draw.text((x1, y1 - 22), label, fill="black", font=font)

    return img


# ═══════════════════════════════════════════════════════════════
# UI COMPONENTS — Individual Gradio component builders
# ═══════════════════════════════════════════════════════════════

def build_header():
    return gr.HTML(f"""
    <div class="app-header">
        <h1>🧠 {APP_TITLE}</h1>
        <p class="subtitle">{APP_SUBTITLE}</p>
        <span class="badge">✦ {MODEL_ACCURACY} accuracy · 5 face shapes · TTA enabled</span>
    </div>
    """)

def build_upload_panel():
    return gr.Image(
        type="pil",
        sources=["upload", "webcam"],
        label="Upload or capture your photo",
        height=420,
        elem_classes=["glass-card"],
    )

def build_results_panel():
    annotated_img = gr.Image(
        label="Detected Face",
        type="pil",
        interactive=False,
        height=300,
        elem_classes=["glass-card"],
    )
    shape_label = gr.HTML(
        value='<div class="prediction-badge"><div style="color:#64748B; font-size:1rem;">Upload a photo to see results</div></div>',
        label="Prediction",
    )
    confidence = gr.Label(
        label="Confidence Scores",
        num_top_classes=5,
        elem_classes=["glass-card"],
    )
    json_out = gr.JSON(label="Full Result (JSON)", visible=True)
    return annotated_img, shape_label, confidence, json_out

def build_shape_info_cards():
    cards_html = '<div class="shape-cards">'
    for name, info in FACE_SHAPE_INFO.items():
        cards_html += f"""
        <div class="shape-card">
            <div class="icon">{info['icon']}</div>
            <div class="name">{name}</div>
            <div class="desc">{info['desc']}</div>
        </div>
        """
    cards_html += '</div>'
    return gr.HTML(cards_html)

def build_examples(image_input):
    if DEMO_DIR.exists():
        example_files = sorted([
            str(f) for f in DEMO_DIR.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ])
        if example_files:
            return gr.Examples(
                examples=example_files,
                inputs=image_input,
                cache_examples=False,
                label="Try an example →",
            )
    return None

def build_footer():
    return gr.HTML(f"""
    <div class="app-footer">
        <strong>Model:</strong> {MODEL_NAME} &nbsp;·&nbsp;
        <strong>Accuracy:</strong> {MODEL_ACCURACY} &nbsp;·&nbsp;
        <strong>Classes:</strong> Heart · Oblong · Oval · Round · Square<br>
        <strong>Stack:</strong> PyTorch Lightning · MediaPipe · EfficientNet-B4 · TTA<br>
        <span style="margin-top: 0.5rem; display: inline-block;">
            Built with 🧠 by Krishna
        </span>
    </div>
    """)


# ═══════════════════════════════════════════════════════════════
# LAYOUT — Full app assembly
# ═══════════════════════════════════════════════════════════════

def build_demo() -> gr.Blocks:
    with gr.Blocks(
        theme=gr.themes.Base(),
        css=CUSTOM_CSS,
        title="Face Shape AI 🧠",
    ) as demo:

        build_header()

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                image_input = build_upload_panel()
                classify_btn = gr.Button(
                    "🔍 Analyze Face Shape",
                    variant="primary",
                    size="lg",
                )
                build_examples(image_input)

            with gr.Column(scale=1):
                annotated_output, shape_label, confidence_bars, json_out = build_results_panel()

        build_shape_info_cards()
        build_footer()

        # ── Wire inference ──
        classify_btn.click(
            fn=run_inference,
            inputs=[image_input],
            outputs=[annotated_output, shape_label, confidence_bars, json_out],
        )
        image_input.change(
            fn=run_inference,
            inputs=[image_input],
            outputs=[annotated_output, shape_label, confidence_bars, json_out],
        )

    return demo


# ═══════════════════════════════════════════════════════════════
# LAUNCH — Entry point
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Face Shape AI — Gradio Demo")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    args = parser.parse_args()

    print()
    print("  ╔══════════════════════════════════════════╗")
    print("  ║        Face Shape AI · Gradio Demo       ║")
    print("  ╚══════════════════════════════════════════╝")
    print(f"  Device : {DEVICE}" +
          (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    print()

    init_model()

    demo = build_demo()
    demo.queue()

    print(f"\n  ✦ Face Shape AI running at: http://localhost:{args.port}")
    if args.share:
        print("  ✦ Generating public share link...")

    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
