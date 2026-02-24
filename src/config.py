import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# Image Preprocessing (PRD Section 4)
IMAGE_SIZE_TRAIN = 224 # EfficientNet-B4 input size (Phase 1/2)
IMAGE_SIZE_INFERENCE = 112
FACE_PADDING = 0.20
ALIGN_EYES = True

# Face Shape Classes (5 Classes - Dropped Diamond/Triangle due to severe starvation)
FACE_SHAPES = ["Heart", "Oblong", "Oval", "Round", "Square"]

# Feature Labels
EYE_SHAPES = ["Almond", "Round", "Hooded", "Monolid", "Downturned", "Upturned"]
NOSE_TYPES = ["Straight", "Concave", "Convex", "Bulbous", "Flat"]
LIP_FULLNESS = ["Thin", "Medium", "Full"]

# Loss Weights (PRD Section 6.2)
LOSS_WEIGHTS = {
    "face_shape_weight": 0.35,
    "landmark_weight": 0.30,
    "features_weight": 0.15,
    "skin_tone_weight": 0.20,
    "focal_gamma": 2.0,
    "label_smoothing": 0.2
}

# Training Hyperparameters (PRD Section 6.3)
EPOCHS = 250
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
BACKBONE_LR_MULTIPLIER = 0.1
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.2

# Scheduler Hyperparameters
WARMUP_EPOCHS = 5
MIN_LR = 1e-7

# Skin Tone Scales
MONK_SCALE = list(range(1, 11))
FITZPATRICK_SCALE = ["I", "II", "III", "IV", "V", "VI"]

# Optimization
ENABLE_FP16 = True
ENABLE_GRADIENT_CLIPPING = True
GRADIENT_CLIP_MAX_NORM = 1.0

def get_config_dict():
    import math
    
    # Calculate derived steps
    real_batch_size = BATCH_SIZE // 2
    grad_accum_steps = 16
    total_samples = 5755
    train_split = 1 - 0.15 - 0.1
    num_train_samples = int(total_samples * train_split)
    
    steps_per_epoch = math.ceil(num_train_samples / (real_batch_size * grad_accum_steps))
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    t_max_steps = (EPOCHS - WARMUP_EPOCHS) * steps_per_epoch
    
    print("=" * 50)
    print("CALCULATED SCHEDULER CONSTANTS:")
    print(f"num_train_samples: {num_train_samples}")
    print(f"steps_per_epoch: {steps_per_epoch}")
    print(f"warmup_steps: {warmup_steps} ({WARMUP_EPOCHS} epochs)")
    print(f"t_max_steps: {t_max_steps} ({EPOCHS - WARMUP_EPOCHS} epochs)")
    print("=" * 50)

    """Returns a dictionary representation of the config for the Lightning trainer."""
    return {
        "project": {
            "seed": 42
        },
        "paths": {
            "processed_data": os.path.join(DATA_DIR, "processed"),
            "landmarks_cache": os.path.join(DATA_DIR, "landmarks_cache"),
            "checkpoints": CHECKPOINT_DIR
        },
        "data": {
            "image_size": IMAGE_SIZE_TRAIN,
            "val_split": 0.15,
            "test_split": 0.1,
            "num_workers": 0,
            "pin_memory": True
        },
        "augmentation": {
            "horizontal_flip_prob": 0.5,
            "rotation_limit": 10,
            "brightness_limit": 0.2,
            "contrast_limit": 0.2,
            "saturation_limit": 0.2,
            "blur_prob": 0.2,
            "noise_prob": 0.2,
            "cutout_prob": 0.1,
            "jpeg_prob": 0.1
        },
        "model": {
            "backbone": "efficientnet_b4",
            "pretrained": True,
            "dropout": 0.4,
            "num_face_shapes": len(FACE_SHAPES),
            "geometric_features": 15
        },
        "loss": LOSS_WEIGHTS,
        "optimizer": {
            "lr": LEARNING_RATE,
            "backbone_lr_multiplier": BACKBONE_LR_MULTIPLIER,
            "weight_decay": WEIGHT_DECAY,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8
        },
        "scheduler": {
            "warmup_epochs": WARMUP_EPOCHS,
            "warmup_steps": warmup_steps,
            "t_max_steps": t_max_steps,
            "min_lr": MIN_LR,
            "steps_per_epoch": steps_per_epoch
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE // 2,
            "mixed_precision": ENABLE_FP16,
            "gradient_accumulation_steps": 16,
            "gradient_clip": GRADIENT_CLIP_MAX_NORM,
            "early_stopping_patience": 15,
            "save_top_k": 3,
            "phases": {
                "unfreeze_full_epoch": 5
            }
        }
    }
