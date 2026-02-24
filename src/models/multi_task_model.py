import torch
import torch.nn as nn
import timm
from src.config import LABEL_SMOOTHING

class MultiTaskFaceModel(nn.Module):
    def __init__(self, backbone_model="efficientnet_b4", pretrained=True):
        super(MultiTaskFaceModel, self).__init__()
        
        # 1. Backbone (PRD Section 5.2 Option A)
        self.backbone = timm.create_model(backbone_model, pretrained=pretrained, num_classes=0, global_pool='avg')
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 380, 380)
            self.feature_dim = self.backbone(dummy_input).shape[1]
            
        print(f"Initialized Backbone: {backbone_model}, Feature Dim: {self.feature_dim}")
        
        # 2. task-Specific Heads (PRD Section 5.4)
        
        # Face Shape Head (7 classes)
        # Input: Flattened backbone features (feature_dim) + 15 geometric ratios
        self.shape_head = nn.Sequential(
            nn.Linear(self.feature_dim + 15, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )
        
        # Feature Analysis Head (Eye Shape, Nose Type, Lip Fullness)
        self.feature_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 6 + 5 + 3)
        )
        
        # Skin Tone Head (ITA Regression + Fitzpatrick/Monk Classification)
        self.skin_head_shared = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU()
        )
        self.ita_branch = nn.Linear(128, 1)                      
        self.fitzpatrick_branch = nn.Linear(128, 6)             
        self.monk_branch = nn.Linear(128, 10)                  

    def forward(self, x, geometric_features=None):
        features = self.backbone(x)
        
        # Shape: Concat backbone features with geometric ratios
        if geometric_features is None:
            # Fallback for inference without explicit math (not recommended for shape task)
            geometric_features = torch.zeros((x.size(0), 15), device=x.device)
            
        shape_input = torch.cat((features, geometric_features), dim=1)
        shape_logits = self.shape_head(shape_input)
        
        # Features
        feature_logits = self.feature_head(features)
        
        # Skin Tone
        skin_shared = self.skin_head_shared(features)
        ita = self.ita_branch(skin_shared)
        fitz = self.fitzpatrick_branch(skin_shared)
        monk = self.monk_branch(skin_shared)
        
        return {
            "shape": shape_logits,
            "features": feature_logits,
            "ita": ita,
            "fitzpatrick": fitz,
            "monk": monk
        }
