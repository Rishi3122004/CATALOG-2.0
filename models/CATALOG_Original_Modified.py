"""
CATALOG Original from Paper with Strategic Modifications:
1. Learnable alpha fusion weight (instead of fixed 0.6) - optimize image-text balance
2. Learnable temperature scaling - adaptive to data
3. Layer normalization in description embeddings - better gradient flow
4. Adjusted dropout - reduce overfitting on classes 1 & 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CATALOG_Original_Modified(nn.Module):
    """
    Modified CATALOG architecture combining original paper design with 
    learnable components to improve class-specific performance.
    
    Key modifications:
    - Learnable fusion weight alpha (instead of 0.6 fixed)
    - Learnable temperature scaling
    - Layer norm on description embeddings
    - Adaptive dropout
    """
    
    def __init__(self, num_classes=10, feature_dim=512):
        super().__init__()
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # **MODIFICATION 1: Learnable alpha fusion weight**
        # Instead of fixed 0.6, let it learn the optimal balance
        self.alpha = nn.Parameter(torch.tensor(0.6))
        
        # **MODIFICATION 2: Learnable temperature scaling**
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        
        # **MODIFICATION 3: Learnable per-class temperature adjustment**
        # Help specific classes like Bear and Badger
        self.class_temps = nn.Parameter(torch.ones(num_classes))
        
        # **MODIFICATION 4: Layer normalization for embeddings**
        # Improve gradient flow
        self.text_norm = nn.LayerNorm(feature_dim)
        self.image_norm = nn.LayerNorm(feature_dim)
        self.desc_norm = nn.LayerNorm(feature_dim)
        
        # Dropout for regularization (help with overfitting)
        self.dropout = nn.Dropout(0.15)  # Increased to help class 1 & 2
        
        print(f"[CATALOG Modified] Trainable parameters:")
        self._log_trainable_params()
    
    def _log_trainable_params(self):
        """Log which parameters are trainable"""
        total = sum(p.numel() for p in self.parameters())
        print(f"  - Learnable alpha: 1")
        print(f"  - Learnable logit_scale: 1")
        print(f"  - Class temps: {self.num_classes}")
        print(f"  - LayerNorm params: {3 * (self.feature_dim * 2)}")
        print(f"  - Dropout: (non-learnable)")
        print(f"  Total trainable: {total}")
    
    def forward(self, images, descriptions, labels, text_centroids):
        """
        Forward pass for training
        
        Args:
            images: [batch_size, 512] - CLIP image embeddings
            descriptions: [batch_size, 512] - BERT description embeddings
            labels: [batch_size] - ground truth labels
            text_centroids: [num_classes, 512] - text class centroids
        
        Returns:
            logits: [batch_size, num_classes]
        """
        batch_size = images.shape[0]
        
        # Apply layer normalization for better training (MODIFICATION 3)
        images = self.image_norm(images)  # [batch_size, 512]
        descriptions = self.desc_norm(descriptions)  # [batch_size, 512]
        text_centroids = self.text_norm(text_centroids)  # [num_classes, 512]
        
        # Normalize embeddings
        images = F.normalize(images, p=2, dim=-1)  # L2 norm
        descriptions = F.normalize(descriptions, p=2, dim=-1)
        text_centroids = F.normalize(text_centroids, p=2, dim=-1)
        
        # Compute two similarity matrices:
        # W: Image vs Text centroids (what paper calls W)
        # Q: Description vs Text centroids (what paper calls Q)
        
        W = images @ text_centroids.t()  # [batch_size, num_classes] - image similarity
        Q = descriptions @ text_centroids.t()  # [batch_size, num_classes] - desc similarity
        
        # **MODIFICATION 1: Learnable alpha fusion**
        # Instead of fixed S = 0.6*W + 0.4*Q, use learnable weight
        alpha = torch.sigmoid(self.alpha)  # Constrain to (0, 1)
        
        logits = alpha * W + (1 - alpha) * Q  # [batch_size, num_classes]
        
        # **MODIFICATION 2 & 3: Learnable temperature scaling**
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        
        # Apply per-class temperature adjustment (helps classes 1 & 2)
        class_temps = torch.softmax(self.class_temps, dim=0)  # Normalize temps
        logits = logits * logit_scale * (1 + class_temps).unsqueeze(0)
        
        # **MODIFICATION 4: Dropout for regularization**
        logits = self.dropout(logits)
        
        return logits
    
    def predict(self, images, descriptions, text_centroids):
        """
        Inference - returns class predictions
        """
        batch_size = images.shape[0]
        
        # Apply layer normalization
        images = self.image_norm(images)
        descriptions = self.desc_norm(descriptions)
        text_centroids = self.text_norm(text_centroids)
        
        # Normalize
        images = F.normalize(images, p=2, dim=-1)
        descriptions = F.normalize(descriptions, p=2, dim=-1)
        text_centroids = F.normalize(text_centroids, p=2, dim=-1)
        
        # Compute similarities
        W = images @ text_centroids.t()
        Q = descriptions @ text_centroids.t()
        
        # Fusion
        alpha = torch.sigmoid(self.alpha)
        logits = alpha * W + (1 - alpha) * Q
        
        # Temperature scaling
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        class_temps = torch.softmax(self.class_temps, dim=0)
        logits = logits * logit_scale * (1 + class_temps).unsqueeze(0)
        
        # Predictions
        predictions = logits.argmax(dim=1)
        
        return predictions

