"""
CATALOG Original from Paper with Strategic Modifications
Modifications:
1. Learnable alpha fusion weight (instead of fixed 0.6)
2. Learnable temperature scaling
3. Proper description projection (768->512)
4. Layer normalization for better training
5. Increased dropout to handle classes 1 & 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CALOGModified(nn.Module):
    """Modified CATALOG with learnable components"""
    
    def __init__(self, num_classes=10, feature_dim=512, desc_dim=768):
        super().__init__()
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # Description projection: 768 -> 1045 -> 512 (paper style MLP)
        self.desc_projection = nn.Sequential(
            nn.Linear(desc_dim, 1045),
            nn.GELU(),
            nn.Linear(1045, feature_dim)
        )
        
        # Learnable fusion weight (MODIFICATION 1)
        self.alpha = nn.Parameter(torch.tensor(0.6))
        
        # Learnable temperature (MODIFICATION 2)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        
        # Learnable per-class temperatures
        self.class_temps = nn.Parameter(torch.ones(num_classes))
        
        # Layer norms (MODIFICATION 4)
        self.image_norm = nn.LayerNorm(feature_dim)
        self.desc_norm = nn.LayerNorm(feature_dim)
        self.text_norm = nn.LayerNorm(feature_dim)
        
        # Dropout (MODIFICATION 5)
        self.dropout = nn.Dropout(0.15)
        
        print(f"[CATALOG Modified] Configuration:")
        print(f"  Feature dim: {feature_dim}")
        print(f"  Desc dim: {desc_dim}")
        print(f"  Num classes: {num_classes}")
        print(f"  Trainable params: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, images, descriptions, labels, text_centroids):
        """
        Forward pass for training
        Arguments:
          images: [B, 512]
          descriptions: [B, 768]
          labels: [B]
          text_centroids: [num_classes, 512]
        """
        # Project descriptions from 768 to 512
        descriptions = self.desc_projection(descriptions)  # [B, 512]
        
        # Apply layer norms
        images = self.image_norm(images)
        descriptions = self.desc_norm(descriptions)
        text_centroids = self.text_norm(text_centroids)
        
        # Normalize
        images = F.normalize(images, p=2, dim=-1)
        descriptions = F.normalize(descriptions, p=2, dim=-1)
        text_centroids = F.normalize(text_centroids, p=2, dim=-1)
        
        # Compute similarities
        W = images @ text_centroids.t()  # [B, num_classes] - image similarity
        Q = descriptions @ text_centroids.t()  # [B, num_classes] - description similarity
        
        # Learnable fusion (MODIFICATION 1)
        alpha = torch.sigmoid(self.alpha)
        logits = alpha * W + (1 - alpha) * Q
        
        # Learnable temperature scaling (MODIFICATION 2)
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        class_temps = torch.softmax(self.class_temps, dim=0)
        logits = logits * logit_scale * (1 + class_temps).unsqueeze(0)
        
        # Dropout
        logits = self.dropout(logits)
        
        return logits
