import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, return_embeds=False):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList()
        self.linears2 = nn.ModuleList()
        self.gelu = QuickGELU()
        self.num_layers = num_layers

        if num_layers > 1:
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            self.linears2.append(nn.Linear(hidden_dim, hidden_dim))

            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
                self.linears2.append(nn.Linear(hidden_dim, hidden_dim, bias=True))

            self.linears.append(nn.Linear(hidden_dim, output_dim, bias=True))
            self.linears2.append(nn.Linear(output_dim, output_dim, bias=True))
        else:
            self.linears.append(nn.Linear(input_dim, output_dim))
            self.linears2.append(nn.Linear(output_dim, output_dim, bias=True))

        self.lns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(output_dim))

        self.softmax = nn.LogSoftmax()
        self.drop = nn.Dropout(dropout)
        self.dropout = dropout
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for lin in self.linears:
            lin.reset_parameters()
        for lin2 in self.linears2:
            lin2.reset_parameters()

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.linears[i](x)
            x = self.lns[i](x)
            x = self.gelu(x)
            x = self.drop(x)
            x = x + self.linears2[i](x)
        return x

class AttentionFusion(nn.Module):
    """Learns attention weights for CLIP and BERT modalities"""
    def __init__(self, feature_dim=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(feature_dim, 512)
        self.key = nn.Linear(feature_dim, 512)
        self.value = nn.Linear(feature_dim, 512)
        self.fc_out = nn.Linear(512, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, clip_feat, bert_feat):
        # Input: [batch, feature_dim]
        # Stack features: [batch, 2, feature_dim]
        combined = torch.stack([clip_feat, bert_feat], dim=1)
        
        # Attention
        Q = self.query(combined)  # [batch, 2, 512]
        K = self.key(combined)
        V = self.value(combined)
        
        attn = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(512)
        attn = F.softmax(attn, dim=-1)  # [batch, 2, 2]
        
        out = torch.matmul(attn, V)  # [batch, 2, 512]
        out = self.fc_out(out)  # [batch, 2, feature_dim]
        
        # Weighted sum
        weights = F.softmax(attn.mean(dim=1), dim=-1)  # [batch, 2]
        fused = (out * weights.unsqueeze(-1)).sum(dim=1)  # [batch, feature_dim]
        
        return fused

class LearnableFusionWeights(nn.Module):
    """Learns fusion weights per sample (adaptive)"""
    def __init__(self, num_features=512):
        super().__init__()
        # Shared weight learner
        self.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # Output: weights for CLIP and BERT
        )
    
    def forward(self, clip_feat, bert_feat):
        # Average features to get context
        combined = (clip_feat + bert_feat) / 2
        
        # Learn weights
        raw_weights = self.fc(combined)
        weights = F.softmax(raw_weights, dim=-1)  # [batch, 2]
        
        # Apply weights
        clip_weight = weights[:, 0:1]  # [batch, 1]
        bert_weight = weights[:, 1:2]
        
        fused = clip_weight * clip_feat + bert_weight * bert_feat
        return fused, weights

class LLaVA_CLIP_Phase2(nn.Module):
    """
    Phase 2 Enhanced CATALOG Model with:
    - Fine-tunable BERT encoder
    - Learnable fusion weights
    - Attention-based fusion
    - Support for hard negative mining
    """
    def __init__(self, hidden_dim, num_layers, dropout, device="", num_classes=10,
                 enable_classifier_fusion=True, fusion_init=-2.2, enable_bert_tuning=True) -> None:
        super().__init__()
        
        # FIX: Project BERT embeddings from 768 → 512 to match CLIP embeddings
        self.description_encoder = nn.Linear(768, 512)
        self.description_encoder_norm = nn.LayerNorm(512)
        
        # Image classifier
        self.image_classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, num_classes)
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enable_classifier_fusion = enable_classifier_fusion
        
        # Phase 2: Learnable components
        self.enable_bert_tuning = enable_bert_tuning
        self.learnable_fusion = LearnableFusionWeights(num_features=512)
        self.attention_fusion = AttentionFusion(feature_dim=512)
        
        # Temperature parameters (now learnable)
        self.logit_scale_CLIP = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_LLaVA = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_fusion = nn.Parameter(torch.ones([]) * np.log(1 / 0.1))  # Dynamic temp
        self.fusion_logit = nn.Parameter(torch.tensor(fusion_init))
        
        # Hard negative mining parameters
        self.margin = 0.3
        self.hard_neg_ratio = 0.5  # Ratio of hard negatives
        
        self.class_weights = None
        self.label_smoothing = 0.03

    def LLaVA_CLIP_loss(self, logits: torch.Tensor, label, t):
        labels = label.to(logits.device).long()
        scaled_logits = logits / t
        if self.class_weights is not None and self.class_weights.numel() == scaled_logits.shape[1]:
            return F.cross_entropy(
                scaled_logits, labels,
                weight=self.class_weights.to(logits.device),
                label_smoothing=self.label_smoothing,
            )
        return F.cross_entropy(scaled_logits, labels, label_smoothing=self.label_smoothing)

    def LLaVA_CLIP_acc(self, logits, target_ind):
        predicted_index = torch.argmax(logits, dim=1)
        acc = torch.sum(predicted_index.cpu() == target_ind.cpu())
        return acc

    def apply_hard_negative_mining(self, similarity, labels, ratio=0.5):
        """Apply hard negative mining to focus on difficult samples"""
        batch_size = similarity.shape[0]
        
        # Find hard examples (low similarity to true class, high to wrong classes)
        labels_onehot = F.one_hot(labels, num_classes=similarity.shape[1]).float()
        true_sim = (similarity * labels_onehot).sum(dim=1)  # True class similarity
        max_wrong_sim = (similarity * (1 - labels_onehot)).max(dim=1)[0]  # Max wrong class
        
        hard_scores = max_wrong_sim - true_sim  # Higher = harder
        
        # Keep all but subsample negatives
        threshold_idx = int(batch_size * ratio)
        _, hard_indices = torch.topk(hard_scores, threshold_idx)
        
        weights = torch.ones(batch_size, device=similarity.device)
        weights[hard_indices] *= 2.0  # Double weight for hard examples
        
        return weights

    def forward(self, embeddings, img_features, txt_features, weight_p, target_ind, temp, use_hard_mining=True):
        
        # Project BERT embeddings from 768 → 512 to match image features
        description_features = embeddings
        if description_features.dim() == 3:  # If [batch, num_classes, 768]
            description_features = description_features.mean(dim=1)
        description_features = self.description_encoder(description_features)  # [batch, 512]
        description_features = self.description_encoder_norm(description_features)
        description_features = description_features / (description_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Attention-based fusion
        fused_visual = self.attention_fusion(img_features, description_features)
        fused_visual = fused_visual / (fused_visual.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Learnable fusion weights (adaptive per sample)
        fused_adaptive, fusion_weights = self.learnable_fusion(img_features, description_features)
        fused_adaptive = fused_adaptive / (fused_adaptive.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Combine attention and learnable fusion (ensemble)
        final_features = (fused_visual + fused_adaptive) / 2
        final_features = final_features / (final_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Compute similarities with learned temperature scaling
        logit_scale_fusion = self.logit_scale_fusion.exp()
        similarity = (final_features @ txt_features) * logit_scale_fusion
        
        # Add classifier fusion
        if self.enable_classifier_fusion:
            cls_logits = self.image_classifier(img_features)
            beta = torch.sigmoid(self.fusion_logit)
            out_logits = (1.0 - beta) * similarity + beta * cls_logits
        else:
            out_logits = similarity
        
        # Compute loss (with or without hard mining)
        if use_hard_mining and self.training:
            hard_weights = self.apply_hard_negative_mining(similarity, target_ind, self.hard_neg_ratio)
            # Store weights in model for weighted loss calculation
            self.current_hard_weights = hard_weights
        else:
            self.current_hard_weights = None
        
        loss = self.LLaVA_CLIP_loss(out_logits, target_ind, temp)
        acc = self.LLaVA_CLIP_acc(out_logits, target_ind)
        
        return loss, acc, torch.argmax(out_logits, dim=1)

    def predict(self, embeddings, img_features, txt_features, weight_p, target_ind, temp):
        description_features = embeddings
        if description_features.dim() == 3:  # If [batch, num_classes, 768]
            description_features = description_features.mean(dim=1)
        description_features = self.description_encoder(description_features)  # [batch, 512]
        description_features = self.description_encoder_norm(description_features)
        description_features = description_features / (description_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        fused_visual = self.attention_fusion(img_features, description_features)
        fused_visual = fused_visual / (fused_visual.norm(dim=-1, keepdim=True) + 1e-8)
        
        fused_adaptive, _ = self.learnable_fusion(img_features, description_features)
        fused_adaptive = fused_adaptive / (fused_adaptive.norm(dim=-1, keepdim=True) + 1e-8)
        
        final_features = (fused_visual + fused_adaptive) / 2
        final_features = final_features / final_features.norm(dim=-1, keepdim=True)
        
        logit_scale_fusion = self.logit_scale_fusion.exp()
        similarity = (final_features @ txt_features) * logit_scale_fusion
        
        if self.enable_classifier_fusion:
            cls_logits = self.image_classifier(img_features)
            beta = torch.sigmoid(self.fusion_logit)
            out_logits = (1.0 - beta) * similarity + beta * cls_logits
        else:
            out_logits = similarity
        
        loss = self.LLaVA_CLIP_loss(out_logits, target_ind, temp)
        acc = self.LLaVA_CLIP_acc(out_logits, target_ind)
        
        return loss, acc, torch.argmax(out_logits, dim=1)

    def accuracy_top_3(self, output, target):
        topk = 3
        pred = output.topk(topk, 1, True, True)[1].t()
        pred = pred.cpu()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = float(correct[:topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        return correct_k
