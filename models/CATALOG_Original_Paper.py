"""
CATALOG: Camera Trap Language-guided Contrastive Learning
Original implementation from paper

Combines multiple Foundation Models (FMs):
- CLIP (image encoder)
- BERT (text encoder from LLaVA descriptions)
- LLaVA (image-text descriptions)
- LLM (text descriptions via templates)

Paper: "CATALOG: Recognizing Camera-Trap Animals in the Wild"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import math


class CATALOG_Original(nn.Module):
    """
    Original CATALOG model from paper.
    
    Architecture:
    1. Text embeddings: LLM descriptions + templates → CLIP text encoder → Centroid
    2. Image embeddings: Images → CLIP image encoder
    3. Image-text embeddings: Images → LLaVA → BERT → MLP (768 → F)
    4. Alignment: Fuse image and image-text embeddings (convex combination)
    5. Loss: Contrastive loss function
    """
    
    def __init__(self, feature_dim=512, num_classes=10, alpha=0.6, temperature=0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.alpha = alpha  # Convex combination parameter
        self.temperature = temperature
        
        # MLP: Project BERT embeddings (768) to feature_dim (512)
        # Paper architecture: single hidden layer
        self.bert_to_feature = nn.Sequential(
            nn.Linear(768, 1045),  # Paper uses 1045 as hidden dim
            nn.GELU(),
            nn.Linear(1045, feature_dim)
        )
        
        # Learnable logit scale (temperature parameter)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.1))
        
    def forward(self, 
                text_embeddings,      # [num_classes, feature_dim] - from CLIP text encoder
                image_embeddings,     # [batch, feature_dim] - from CLIP image encoder  
                image_text_embeddings, # [batch, 768] - from BERT (LLaVA descriptions)
                target_labels):       # [batch] - ground truth class indices
        """
        Forward pass for CATALOG.
        
        Args:
            text_embeddings: Class embeddings from CLIP text encoder (centroid of descriptions)
            image_embeddings: Image features from CLIP image encoder
            image_text_embeddings: Image description embeddings from BERT
            target_labels: Ground truth labels
            
        Returns:
            loss: Contrastive loss
            accuracy: Batch accuracy
        """
        
        batch_size = image_embeddings.shape[0]
        
        # Step 1: Project image-text embeddings (BERT 768 → feature_dim 512)
        image_text_proj = self.bert_to_feature(image_text_embeddings)  # [batch, 512]
        
        # Step 2: Normalize embeddings L2
        image_embeddings_norm = F.normalize(image_embeddings, p=2, dim=-1)  # [batch, 512]
        image_text_proj_norm = F.normalize(image_text_proj, p=2, dim=-1)   # [batch, 512]
        text_embeddings_norm = F.normalize(text_embeddings, p=2, dim=-1)   # [num_classes, 512]
        
        # Step 3: Compute similarity matrices
        # W: cosine similarity between image and text embeddings
        # text_embeddings_norm is [num_classes, 512]
        W = image_embeddings_norm @ text_embeddings_norm.t()  # [batch, 512] @ [512, num_classes] = [batch, num_classes]
        
        # Q: cosine similarity between image-text and text embeddings
        Q = image_text_proj_norm @ text_embeddings_norm.t()   # [batch, 512] @ [512, num_classes] = [batch, num_classes]
        
        # Step 4: Fusion mechanism (Equation 4 from paper)
        S = self.alpha * W + (1 - self.alpha) * Q  # [batch, num_classes]
        
        # Step 5: Contrastive loss (Equation 5 from paper)
        # Scale by learnable temperature
        logits = S / torch.exp(self.logit_scale)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, target_labels)
        
        # Compute accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == target_labels).float().mean()
        
        return loss, accuracy.item(), predictions
    
    def predict(self,
                text_embeddings,
                image_embeddings,
                image_text_embeddings):
        """
        Prediction only (no loss computation).
        """
        batch_size = image_embeddings.shape[0]
        
        # Project and normalize
        image_text_proj = self.bert_to_feature(image_text_embeddings)
        
        image_embeddings_norm = F.normalize(image_embeddings, p=2, dim=-1)
        image_text_proj_norm = F.normalize(image_text_proj, p=2, dim=-1)
        text_embeddings_norm = F.normalize(text_embeddings, p=2, dim=-1)
        
        # Compute similarities
        W = image_embeddings_norm @ text_embeddings_norm.t()
        Q = image_text_proj_norm @ text_embeddings_norm.t()
        
        # Fusion
        S = self.alpha * W + (1 - self.alpha) * Q
        
        # Scale and predict
        logits = S / torch.exp(self.logit_scale)
        predictions = torch.argmax(logits, dim=1)
        
        return logits, predictions
