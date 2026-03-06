# Face Analysis AI (V6 Multi-Task Model)

A robust, multi-task AI model for deep facial analysis. Built on an `EfficientNet-B4` backbone, this V6 model features **8 independent prediction heads** yielding high-accuracy classification for face shape, facial attributes, demographic indicators, and quantitative geometry.

## 🚀 Key Features

* **Multi-Task Architecture:** End-to-end `EfficientNet-B4` pipeline passing visual features + structural (geometric) features + color features to customized heads.
* **8 Prediction Heads:**
  * **Face Shape (5-class):** Heart, Oblong, Oval, Round, Square (79.2% Test Accuracy)
  * **Skin Tone (3-class):** Light, Medium, Dark (82.3% Test Accuracy) using a Two-Tower HSV feature fusion.
  * **Facial Attributes:** Eye Shape (Narrow/Almond/Big), Brow Type, Lip Shape.
  * **Demographics:** Age Group (Young/Older), Gender.
  * **Geometry:** 15-point spatial relationship prediction bounding facial landmarks.
* **Premium Gradio UI:** A stunning, dark-themed glassmorphism interface (`app.py`) designed for complete analysis and visualizations.
* **ONNX CPU Acceleration:** `export_onnx.py` provided to package the PyTorch Lightning model into a fast graph suitable for edge environments.

## 📁 Project Structure

```text
├── src/
│   ├── data/           # MediaPipe augmentations & HSV extraction classes
│   ├── models/         # EfficientNet-B4 multi-task & two-tower architecture
│   └── training/       # PyTorch Lightning modules for focal loss & class weights
├── app.py              # Premium Gradio web demo
├── predict.py          # Unified inference script & programmatic prediction API
├── export_onnx.py      # ONNX export utility for production deployment
├── train_attributes_v2.py # Head-specific freezing training script
├── train_multitask_v3.py  # Full-model training script
└── COLAB_GUIDE.md      # Checklist for cloud T4 GPU training
```

## 🛠️ Installation

```powershell
# Create and activate virtual environment (Python 3.10 recommended)
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚦 Usage

### 1. Run the Web Interface (Gradio)

The web demo is the recommended way to use the model, supporting image uploads and providing detailed confidence charts:

```powershell
python app.py
```

*It will automatically launch a server at `http://localhost:7860`.*

### 2. Command-Line Inference

Process a single image or an entire folder directly from the command line:

```powershell
python predict.py --image path/to/face.jpg --checkpoint checkpoints/attributes_v2/last-v5.ckpt
```

*Returns a detailed JSON object containing `predicted_class`, `skin_tone`, `eye_shape`, and more.*

### 3. ONNX Export

To prepare the model for C++ / C# or lightweight container deployment, export it to ONNX:

```powershell
python export_onnx.py --ckpt checkpoints/attributes_v2/last-v5.ckpt --out checkpoints/attributes_v2/model_v6.onnx
```

## 📊 V6 Model Details

**Skin Tone Integration**: The V6 module bypasses early-layer color dropping in EfficientNet by calculating a 48-bin `HSV Histogram` from the face crop and feeding it directly into a discrete `SkinTower` appended to the feature map. Minority classes (e.g. Dark tones) are heavily oversampled and balanced with `BCEWithLogitsLoss(pos_weight)`.

**Optimization**: The final attributes tune involved freezing the `EfficientNet` backbone and Face Shape head entirely via `requires_grad = False` to prevent face shape regression while the new attribute heads converged.

---
*Built with PyTorch Lightning and Gradio.*
