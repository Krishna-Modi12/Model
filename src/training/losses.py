import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import LOSS_WEIGHTS, LABEL_SMOOTHING

class MultiTaskLoss(nn.Module):
    def __init__(self, weights=LOSS_WEIGHTS):
        super(MultiTaskLoss, self).__init__()
        self.weights = weights
        
        # CrossEntropy with Label Smoothing (Section 6.5)
        self.shape_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        
        # Binary Cross Entropy for multi-label features
        self.feature_criterion = nn.BCEWithLogitsLoss()
        
        # Skin Tone losses
        self.ita_criterion = nn.MSELoss()
        self.skin_scale_criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        """
        outputs: dict from MultiTaskFaceModel
        targets: dict from UnifiedFaceDataset
        """
        # 1. Face Shape Loss
        l_shape = self.shape_criterion(outputs['shape'], targets['shape'])
        
        # 2. Features Loss (Eye, Nose, Lips)
        # Split outputs: [Eye(6), Nose(5), Lips(3)]
        eye_logits = outputs['features'][:, :6]
        nose_logits = outputs['features'][:, 6:11]
        lip_logits = outputs['features'][:, 11:]
        
        # Targets are long indices: [Eye_Idx, Nose_Idx, Lip_Idx]
        l_eye = self.skin_scale_criterion(eye_logits, targets['features'][:, 0].long())
        l_nose = self.skin_scale_criterion(nose_logits, targets['features'][:, 1].long())
        l_lip = self.skin_scale_criterion(lip_logits, targets['features'][:, 2].long())
        
        l_features = (l_eye + l_nose + l_lip) / 3.0
        
        # 3. Skin Tone Loss
        l_ita = self.ita_criterion(outputs['ita'].squeeze(), targets['ita'])
        l_fitz = self.skin_scale_criterion(outputs['fitzpatrick'], targets['skin_scale'][:, 0])
        l_monk = self.skin_scale_criterion(outputs['monk'], targets['skin_scale'][:, 1])
        
        l_skin = l_ita + (0.5 * l_fitz) + (0.5 * l_monk)
        
        # 4. Total Weighted Loss (PRD Section 6.2)
        # We skip landmark loss (0.30) for now and re-distribute or keep it for future heatmaps
        # redistributed_weights = shape: 0.50, features: 0.20, skin: 0.30
        
        total_loss = (self.weights['face_shape_weight'] * l_shape) + \
                     (self.weights['features_weight'] * l_features) + \
                     (self.weights['skin_tone_weight'] * l_skin)
        
        return {
            "total": total_loss,
            "shape": l_shape,
            "features": l_features,
            "skin": l_skin
        }
