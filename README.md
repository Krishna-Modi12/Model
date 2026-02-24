# Face Analysis AI

A comprehensive multi-task AI model for face analysis, including face shape classification, facial feature analysis, and skin tone estimation.

## 🚀 Key Features

* **Multi-Task Architecture**: EfficientNet-B4 backbone with specialized heads.
  * **Face Shape**: 7-class classification (Oval, Round, Square, etc.) using hybrid deep features + 15 spatial ratios.
  * **Facial Features**: Multi-label detection for eyes, nose, and lips.
  * **Skin Tone**: ITA calculation in LAB space with mapping to Monk and Fitzpatrick scales.
* **Production-Ready Pipeline**:
  * Pre-processing with MediaPipe landmark alignment and CLAHE enhancement.
  * Automated data curation tool to filter blurs and pre-label skin tones.
  * Ethics auditing for demographic balance.
  * Comprehensive reporting with confusion matrices and regression plots.

## 📁 Project Structure

```text
├── src/
│   ├── data/           # Dataset loading & preprocessing
│   ├── models/         # Model architecture
│   ├── training/       # Training loops & loss functions
│   └── utils/          # Geometry, skin utils, ethics, and reports
├── data/               # Project data storage
├── checkpoints/        # Model checkpoints
├── curate_data.py      # Automated data labeling utility
├── train.py            # Main training entry point
└── requirements.txt    # dependencies
```

## 🛠️ Installation

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚦 Usage

### 1. Curate your Dataset

Use the curation tool to filter your raw images and generate a labeling template with pre-computed skin tones:

```bash
python curate_data.py --img_dir path/to/raw_images --output data/labels.csv
```

### 2. Verify Architecture

Run a dry run on your GPU to ensure the entire pipeline is functional:

```bash
python train.py --dry_run
```

### 3. Start Training

Once your labels are ready:

```bash
python train.py --train_csv data/train.csv --image_dir data/images
```

## 📊 Monitoring

Experiments are tracked via MLflow. View the dashboard:

```bash
mlflow ui
```

## 🔗 Specifications

This project adheres to the **PRD v2.2.0** specifications for Face Analysis AI.
