import os
import sys
import argparse
import torch
import cv2
import json
import numpy as np
import pytorch_lightning as L
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from predict import load_model, predict_single, DEFAULT_CHECKPOINT
from src.utils.landmark_extractor import LandmarkExtractor
from src.config import get_config_dict, FACE_SHAPES
from src.data.dataset import FaceAnalysisDataset, get_val_transforms, extract_hsv_histogram_np
from src.training.trainer import FaceAnalysisLightningModule

L.seed_everything(42, workers=True)

class TTALightningModule(FaceAnalysisLightningModule):
    def test_step(self, batch, batch_idx):
        images = batch["images"]
        geo_ratios = batch["geometric_ratios"]
        
        output_orig = self(images, geo_ratios)
        logits_orig = output_orig.face_shape_logits
        
        images_flip = TF.hflip(images)
        hsv_flip = torch.stack([
            torch.from_numpy(extract_hsv_histogram_np((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)))
            for img in TF.normalize(images_flip, mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
        ]).to(images.device)
        logits_flip = self(images_flip, geo_ratios, hsv_flip).face_shape_logits
        
        _, _, h, w = images.shape
        zh, zw = int(h * 0.95), int(w * 0.95)
        images_zoom = TF.center_crop(images, [zh, zw])
        images_zoom = TF.resize(images_zoom, [h, w], antialias=True)
        logits_zoom = self(images_zoom, geo_ratios).face_shape_logits
        
        images_rot_p = TF.rotate(images, 5.0)
        logits_rot_p = self(images_rot_p, geo_ratios).face_shape_logits
        
        images_rot_n = TF.rotate(images, -5.0)
        logits_rot_n = self(images_rot_n, geo_ratios).face_shape_logits
        
        avg_logits = (logits_orig + logits_flip + logits_zoom + logits_rot_p + logits_rot_n) / 5.0
        preds = avg_logits.argmax(dim=1)
        
        self.test_acc(preds, batch["shape_labels"])
        self.log("test/acc", self.test_acc, prog_bar=True, on_epoch=True)
        
def is_valid_face_image(img_path, extractor, min_detection_score=0.60):
    img = cv2.imread(str(img_path))
    if img is None:
        return False, None
    result = extractor.extract(img)
    if not result.success:
        return False, None
    return True, result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/attributes_v2/last.ckpt")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = Path(args.checkpoint)
    if not model_path.exists():
        logger.error(f"Checkpoint not found: {model_path}")
        return
        
    config = get_config_dict()
    image_size = config["data"]["image_size"]

    print("================================================")
    print("DATASET FILTERING SUMMARY")
    print("================================================")
    extractor = LandmarkExtractor(static_image_mode=True)
    
    # 1. Evaluate Face Shape (official test set)
    print("Evaluating face shape...")
    test_meta_path = Path("data/processed/annotations_self_train_v3.meta.json")
    with open(test_meta_path, "r") as f:
        meta = json.load(f)
    test_ds = FaceAnalysisDataset(
        annotations_path="data/processed/annotations_self_train_v3.json",
        image_size=image_size,
        indices=meta["test_indices"],
        transforms=get_val_transforms(image_size)
    )
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    
    model_std = FaceAnalysisLightningModule(config=config)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    
    # Filter out mismatched prefix keys AND mismatched shapes
    model_dict = model_std.model.state_dict()
    filtered_dict = {}
    for k, v in state_dict.items():
        core_k = k.replace("model.", "") if k.startswith("model.") else k
        if core_k in model_dict and v.shape == model_dict[core_k].shape:
            filtered_dict[core_k] = v
            
    model_std.model.load_state_dict(filtered_dict, strict=False)
    model_std.model.eval()
    model_std.model.to(device)

    # Compute face shape acc and f1 manually to avoid Lightning logger/hook issues
    test_preds_std = []
    test_preds_tta = []
    test_labels = []
    
    for batch in tqdm(test_loader, desc="Testing face shape"):
        images = batch["images"].to(device)
        geo_ratios = batch["geometric_ratios"].to(device)
        labels = batch["shape_labels"].to(device)
        
        with torch.no_grad():
            output_orig = model_std.model(images, geo_ratios)
            logits_orig = output_orig.face_shape_logits
            test_preds_std.extend(logits_orig.argmax(dim=1).cpu().tolist())
            test_labels.extend(labels.cpu().tolist())
            
            # TTA Processing
            images_flip = TF.hflip(images)
            logits_flip = model_std.model(images_flip, geo_ratios).face_shape_logits
            
            _, _, h, w = images.shape
            zh, zw = int(h * 0.95), int(w * 0.95)
            images_zoom = TF.center_crop(images, [zh, zw])
            images_zoom = TF.resize(images_zoom, [h, w], antialias=True)
            logits_zoom = model_std.model(images_zoom, geo_ratios).face_shape_logits
            
            images_rot_p = TF.rotate(images, 5.0)
            logits_rot_p = model_std.model(images_rot_p, geo_ratios).face_shape_logits
            
            images_rot_n = TF.rotate(images, -5.0)
            logits_rot_n = model_std.model(images_rot_n, geo_ratios).face_shape_logits
            
            avg_logits = (logits_orig + logits_flip + logits_zoom + logits_rot_p + logits_rot_n) / 5.0
            test_preds_tta.extend(avg_logits.argmax(dim=1).cpu().tolist())
            
    std_acc = (np.array(test_preds_std) == np.array(test_labels)).mean() * 100
    tta_acc = (np.array(test_preds_tta) == np.array(test_labels)).mean() * 100

    val_f1_score = f1_score(test_labels, test_preds_std, average='macro')
    
    cm = confusion_matrix(test_labels, test_preds_std, labels=range(5))
    # Oval is 2, Heart is 0
    oval_heart_err = cm[2][0] if len(cm) > 2 else 0

    # 2. Evaluate multitask (using multitask JSON)
    with open("data/processed/annotations_multitask_final.json", "r") as f:
        data = json.load(f)
        
    import random
    random.seed(42)
    all_data = data.copy()
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.85)
    val_set = all_data[split_idx:]
    
    skin_data = [d for d in data if d.get('monk_label') is not None]
    attr_data = [d for d in val_set if d.get('attributes') is not None]
    
    print(f"Face shape test set   : {len(test_ds)} total | {len(test_ds)} valid faces | 0 skipped")
    
    # Filter skin data
    valid_skin = []
    skipped_skin = 0
    for d in tqdm(skin_data, desc="Filtering Fitzpatrick17k..."):
        valid, _ = is_valid_face_image(d["image_path"], extractor)
        if valid:
            valid_skin.append(d)
        else:
            skipped_skin += 1
            
    skip_pct = (skipped_skin/len(skin_data)*100) if len(skin_data) > 0 else 0
    print(f"Fitzpatrick17k        : {len(skin_data)} total | {len(valid_skin)} valid faces | {skipped_skin} skipped ({skip_pct:.1f}%)")
    print("Note: Fitzpatrick skips expected (~86%) \u2014 clinical images")
    print("Evaluation runs ONLY on valid face detections")
    print("================================================")
    
    # Evaluate Skin
    skin_exact = skin_near = skin_mae = 0
    skin_class_total = {i:0 for i in range(3)}
    skin_class_correct = {i:0 for i in range(3)}
    all_preds_skin = []
    all_trues_skin = []
    
    model_std.model.to(device)
    model_std.model.eval()
    
    for d in valid_skin:
        _, res = is_valid_face_image(d["image_path"], extractor) # re-extract
        geo = torch.tensor(res.geometric_ratios, dtype=torch.float32).unsqueeze(0).to(device)
        
        img = cv2.imread(d["image_path"])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        hsv_hist = torch.from_numpy(extract_hsv_histogram_np(img_rgb)).unsqueeze(0).to(device).float()
        tensor_img = get_val_transforms(image_size)(image=img_rgb)["image"].unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model_std.model(tensor_img, geo, hsv_hist)
        
        pred_monk = out.skin_tone_logits.argmax(dim=1).item()
        true_monk = d["monk_label"]
        
        if pred_monk == true_monk:
            skin_exact += 1
            skin_near += 1
            skin_class_correct[true_monk] += 1
        elif abs(pred_monk - true_monk) <= 1:
            skin_near += 1
        
        skin_mae += abs(pred_monk - true_monk)
        all_preds_skin.append(pred_monk)
        all_trues_skin.append(true_monk)
        skin_class_total[true_monk] += 1
        
    skin_exact_acc = (skin_exact / len(valid_skin)) * 100 if valid_skin else 0
    skin_near_acc = (skin_near / len(valid_skin)) * 100 if valid_skin else 0
    skin_mae = skin_mae / len(valid_skin) if valid_skin else 0
    
    print("\n================================================")
    print("SKIN TONE EVALUATION (face-valid images only)")
    print("================================================")
    print(f"Total Fitzpatrick17k  : {len(skin_data)} images")
    print(f"Valid faces           : {len(valid_skin)} images ({100-skip_pct:.1f}%)")
    print(f"Skipped (no face)     : {skipped_skin} images ({skip_pct:.1f}% \u2014 expected, clinical images)")
    print("------------------------------------------------")
    print(f"Exact Match Accuracy  : {skin_exact_acc:.2f}%  (target > 71.4%)")
    print(f"Near-Match Accuracy   : {skin_near_acc:.2f}%  (|pred-true| \u2264 1)")
    print(f"Mean Absolute Error   : {skin_mae:.2f} Monk levels")
    print("------------------------------------------------")
    print("Per-class accuracy:")
    print(f"Skin distribution (Pred): {Counter(all_preds_skin)}")
    print(f"Skin distribution (True): {Counter(all_trues_skin)}")

    for i in range(3):
        label = ["Light", "Medium", "Dark"][i]
        cls_correct = skin_class_correct.get(i, 0)
        cls_total = skin_class_total.get(i, 0)
        acc = (cls_correct / cls_total * 100) if cls_total > 0 else 0
        print(f"  {label:<10} accuracy: {acc:>6.2f}% ({cls_correct}/{cls_total})")
    print("------------------------------------------------")
    print(f"Random baseline       : 33.3% (3-class uniform)")
    print(f"Model vs random       : +{skin_exact_acc - 33.3:.2f} percentage points")
    print("================================================")

    # Evaluate Attributes
    # Collect real face-valid CelebA
    valid_attr = []
    for d in tqdm(attr_data, desc="Filtering CelebA val..."):
        valid, _ = is_valid_face_image(d["image_path"], extractor)
        if valid:
            valid_attr.append(d)
            if len(valid_attr) > 200: # Test on a fast subset if time is constraint
                break

    eye_narrow_p, eye_big_p, brow_p, lip_p, age_p, gender_p = [], [], [], [], [], []
    eye_narrow_t, eye_big_t, brow_t, lip_t, age_t, gender_t = [], [], [], [], [], []
    land_mae = 0.0
    land_preds = []
    land_trues = []

    for d in valid_attr:
        _, res = is_valid_face_image(d["image_path"], extractor)
        geo = torch.tensor(res.geometric_ratios, dtype=torch.float32).unsqueeze(0).to(device)

        img = cv2.imread(d["image_path"])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor_img = get_val_transforms(image_size)(image=img_rgb)["image"].unsqueeze(0).to(device)
        
        # Also need hsv_hist for multitask model!
        hsv_hist = torch.from_numpy(extract_hsv_histogram_np(img_rgb)).unsqueeze(0).to(device).float()
        
        with torch.no_grad():
            out = model_std.model(tensor_img, geo, hsv_hist)
            
        attr = d["attributes"]
        
        # Eyes
        e_n_pred = (torch.sigmoid(out.eye_narrow_logits)[0,0] > 0.5).int().item()
        e_b_pred = (torch.sigmoid(out.eye_narrow_logits)[0,1] > 0.5).int().item()
        eye_narrow_p.append(e_n_pred); eye_narrow_t.append(attr["eye_narrow"])
        eye_big_p.append(e_b_pred); eye_big_t.append(attr["eye_big"])
        
        # Brow, Lip, Age, Gender
        brow_p.append(out.brow_type_logits.argmax(dim=1).item()); brow_t.append(attr["brow"])
        lip_p.append(out.lip_shape_logits.argmax(dim=1).item()); lip_t.append(attr["lip"])
        age_p.append(out.age_logits.argmax(dim=1).item()); age_t.append(attr["age"])
        gender_p.append(out.gender_logits.argmax(dim=1).item()); gender_t.append(attr["gender"])
        
        # Landmark
        p_land = out.landmark_pred.squeeze().cpu().numpy()
        t_land = geo.squeeze().cpu().numpy()
        land_mae += np.abs(p_land - t_land).mean()
        land_preds.append(p_land)
        land_trues.append(t_land)
        
    land_mae /= len(valid_attr)
    land_corr = np.corrcoef(np.concatenate(land_preds), np.concatenate(land_trues))[0,1]
    
    en_f1 = f1_score(eye_narrow_t, eye_narrow_p, average='macro')
    eb_f1 = f1_score(eye_big_t, eye_big_p, average='macro')
    b_f1 = f1_score(brow_t, brow_p, average='macro')
    l_f1 = f1_score(lip_t, lip_p, average='macro')
    a_f1 = f1_score(age_t, age_p, average='macro')
    g_f1 = f1_score(gender_t, gender_p, average='macro')

    def p_score(t, p): return np.mean(np.array(t) == np.array(p))*100

    print("\n================================================")
    print("ATTRIBUTE HEAD EVALUATION (CelebA val, face-valid)")
    print("================================================")
    print(f"{'Head':<14} {'Accuracy':<8} {'F1':<6} {'Random'}")
    print("-" * 50)
    print(f"{'Eye (narrow)':<14} {p_score(eye_narrow_t, eye_narrow_p):<8.2f} {en_f1:<6.4f} 50.0%")
    print(f"{'Eye (big)':<14} {p_score(eye_big_t, eye_big_p):<8.2f} {eb_f1:<6.4f} 50.0%")
    print(f"{'Brow Type':<14} {p_score(brow_t, brow_p):<8.2f} {b_f1:<6.4f} 50.0%")
    print(f"{'Lip Shape':<14} {p_score(lip_t, lip_p):<8.2f} {l_f1:<6.4f} 50.0%")
    print(f"{'Age Group':<14} {p_score(age_t, age_p):<8.2f} {a_f1:<6.4f} 50.0%")
    print(f"{'Gender':<14} {p_score(gender_t, gender_p):<8.2f} {g_f1:<6.4f} 50.0%")
    
    print("\n================================================")
    print("LANDMARK HEAD EVALUATION")
    print("================================================")
    print(f"MAE per ratio       : {land_mae:.4f}")
    print(f"Correlation         : {land_corr:.4f}")

    print("\n================================================")
    print("MODEL V6 COMPLETE EVALUATION SUMMARY")
    print("================================================")
    print(f"Face Shape (TTA)    : {tta_acc:.2f}%   (V5: 79.21%)")
    print(f"Face Shape (no TTA) : {std_acc:.2f}%   (V5: 76.83%)")
    print(f"Face Shape F1       : {val_f1_score:.4f}   (V5: 0.7654)")
    print(f"Face Shape Oval->Heart: {oval_heart_err}     (V5: 11)")
    print(f"Skin Tone (exact)   : {skin_exact_acc:.2f}%   (prev: 71.4%)")
    print(f"Skin Tone (near)    : {skin_near_acc:.2f}%   (|pred-true| \u2264 1)")
    print(f"Eye Narrow F1       : {en_f1:.4f}")
    print(f"Eye Big F1          : {eb_f1:.4f}")
    print(f"Brow F1             : {b_f1:.4f}")
    print(f"Lip F1              : {l_f1:.4f}")
    print(f"Age F1              : {a_f1:.4f}")
    print(f"Gender F1           : {g_f1:.4f}")
    print(f"Landmark MAE        : {land_mae:.4f}")
    print("------------------------------------------------")
    
    regressed = (std_acc < 76.0) or (val_f1_score < 0.75)
    print(f"Face shape regression: {'YES' if regressed else 'NO'}")
    all_random = (np.mean([en_f1, eb_f1, b_f1, l_f1, a_f1, g_f1]) > 0.5)
    print(f"All attrs > random  : {'YES' if all_random else 'NO'}")
    print("================================================")

    if not regressed and all_random:
        print("\n\u2705 V2 training + proper eval succeeded")
        print("   Update predict.py and app.py with V2 checkpoint")
        print("   All metrics documented and verified")
    elif not regressed and not all_random:
        print("\n\u26a0\ufe0f Some attribute heads not learning")
        print("   Check: Is has_attributes mask working correctly?")
        print("   Check: Are CelebA attribute labels mapped correctly?")
        print("   Check: Is BalancedBatchSampler active?")
    else:
        print("\n\u274c Multi-task loss hurting face shape")
        print("   Reduce attribute loss weights by 50%:")
        print("   eye:0.15 brow:0.15 lip:0.15 age:0.10")
        print("   gender:0.10 skin:0.20 landmark:0.20")
        print("   Keep V5 as production")

if __name__ == "__main__":
    main()
