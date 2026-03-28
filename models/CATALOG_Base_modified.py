import math

import torch.nn as nn
import torch
import numpy as np

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):
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
        # The log softmax layer
        self.softmax = nn.LogSoftmax()
        self.drop = nn.Dropout(dropout)
        self.dropout = dropout
        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for lin in self.linears:
            lin.reset_parameters()
        for lin2 in self.linears2:
            lin2.reset_parameters()
        for bn in self.lns:
            bn.reset_parameters()

    def forward(self, x):
        if self.num_layers > 1:
            for i, (lin, lin2) in enumerate( zip(self.linears[:-1], self.linears2[:-1])):
                embed1 = lin(x)
                embed2 = self.drop(lin2(self.gelu(embed1)))
                x = self.lns[i](embed1 + embed2)


            embed1 = self.linears[-1](x)
            embed2 = self.drop(self.linears2[-1](self.gelu(embed1)))
            x = self.lns[-1](embed1 + embed2)
        else:
            embed1 = self.linears[0](x)
            embed2 = self.drop(self.linears2[0](self.gelu(embed1)))
            x = self.lns[0](embed1 + embed2)

        if self.return_embeds:
            out = x
        else:
            out = self.softmax(x)

        return out


class LLaVA_CLIP_modified(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout, device="") -> None:
        super().__init__()
        # Bypass BERT and MLP: no description_encoder
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # temperature
        self.logit_scale_CLIP = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # Dynamic alpha: learnable parameter
        self.alpha_param = nn.Parameter(torch.tensor(0.5))

    def get_alpha(self):
        return torch.sigmoid(self.alpha_param)  # 0-1

    def _compute_logits(self, img_features, txt_features):
        # Dynamic alpha scales CLIP similarity contribution.
        alpha = self.get_alpha()
        logit_scale_CLIP = self.logit_scale_CLIP.exp()
        similarity = (img_features @ txt_features) * logit_scale_CLIP
        return alpha * similarity

    # Loss function modification: add entropy regularization
    def LLaVA_CLIP_loss_modified(self, logits: torch.Tensor, label, t):
        # Base loss
        loss = self.LLaVA_CLIP_loss(logits, label, t)
        # Add entropy to encourage diversity
        probs = torch.softmax(logits / t, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
        return loss + 0.01 * entropy

    def LLaVA_CLIP_loss(self, logits: torch.Tensor, label, t):
        loss_i = 0
        for b in range(len(logits)):
            num = torch.exp(logits[b][label[b]] / t)
            dem = torch.sum(torch.exp(logits[b] / t))
            loss_i += torch.log(num / dem)
        loss = -loss_i / len(logits)
        return loss

    def LLaVA_CLIP_acc(self, img_feat, text_feat, target_ind):
        logits = self._compute_logits(img_feat, text_feat)
        predicted_index = torch.argmax(logits, dim=1)
        acc = torch.sum(predicted_index.cpu() == target_ind)
        return acc

    def forward(self, img_features, txt_features, target_ind, temp):
        # Bypass: only CLIP
        out_logits = self._compute_logits(img_features, txt_features)

        loss = self.LLaVA_CLIP_loss_modified(out_logits, target_ind, temp)
        acc = self.LLaVA_CLIP_acc(img_features, txt_features, target_ind)
        return loss, acc, torch.argmax(out_logits, dim=1)

    def predict(self, img_features, txt_features, target_ind, temp):
        out_logits = self._compute_logits(img_features, txt_features)
        max_values, max_indices = torch.max(out_logits, dim=1)
        return max_values, max_indices

    def predict_top_3(self, img_features, txt_features, target_ind, temp):
        out_logits = self._compute_logits(img_features, txt_features)
        topk = 3
        pred = out_logits.topk(topk, 1, True, True)[1].t()
        return pred