# ☁️ Training on Google Colab Guide

If your local PC is unstable or you want to speed up training using free cloud GPUs (like the NVIDIA T4), follow these steps:

### 1. Prepare your Project

Ensure you have the following files in your project root:

- `requirements.txt`
- `train_colab.ipynb`
- `annotations.json`
- `src/` directory

### 2. Upload to Google Drive

1. Go to [Google Drive](https://drive.google.com/).
2. Create a folder named `Model` (or any name you prefer).
3. Upload your entire project directory into this folder.
   > **Tip:** I have already created **`FaceShapeAI_Project.zip`** in your project root! You can just upload that single file to your Drive for maximum speed.

### 3. Open in Colab

1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **File** > **Upload Notebook** and select `train_colab.ipynb` from your local machine.

### 4. Configure & Train

1. **Runtime:** Click **Runtime** > **Change runtime type** > **Hardware accelerator** > **T4 GPU**.
2. **Mount Drive:** Run the first cell to connect Colab to your Google Drive.
3. **Set Path:** In the second code cell, update `PROJECT_PATH` to match your Drive folder path (e.g., `/content/drive/MyDrive/Model`).
4. **Install:** Run the install cell.
5. **Train:** Run the `!python phase4_convnext.py` cell to start training!

### 5. Benefits

- **Stability:** Colab won't restart your PC.
- **Speed:** T4 GPUs are often faster for small-to-medium models.
- **Persistence:** Save your checkpoints directly back to Google Drive.
