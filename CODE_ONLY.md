# CATALOG MODIFIED - CODE ONLY

---

## 1. LEARNABLE ALPHA

```python
self.alpha = nn.Parameter(torch.tensor(0.6))

alpha = torch.sigmoid(self.alpha)
logits = alpha * W + (1 - alpha) * Q
```

---

## 2. LEARNABLE TEMPERATURE

```python
self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
self.class_temps = nn.Parameter(torch.ones(num_classes))

logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
class_temps = torch.softmax(self.class_temps, dim=0)
logits = logits * logit_scale * (1 + class_temps).unsqueeze(0)
```

---

## 3. ENHANCED MLP PROJECTION

```python
self.desc_projection = nn.Sequential(
    nn.Linear(desc_dim, 1045),
    nn.GELU(),
    nn.Linear(1045, feature_dim)
)

descriptions = self.desc_projection(descriptions)
```

---

## 4. LAYER NORMALIZATION

```python
self.image_norm = nn.LayerNorm(feature_dim)
self.desc_norm = nn.LayerNorm(feature_dim)
self.text_norm = nn.LayerNorm(feature_dim)

images = self.image_norm(images)
descriptions = self.desc_norm(descriptions)
text_centroids = self.text_norm(text_centroids)
```

---

## 5. DROPOUT

```python
self.dropout = nn.Dropout(0.15)

logits = self.dropout(logits)
return logits
```

---

## FORWARD PASS

```python
def forward(self, images, descriptions, labels, text_centroids):
    descriptions = self.desc_projection(descriptions)
    
    images = self.image_norm(images)
    descriptions = self.desc_norm(descriptions)
    text_centroids = self.text_norm(text_centroids)
    
    images = F.normalize(images, p=2, dim=-1)
    descriptions = F.normalize(descriptions, p=2, dim=-1)
    text_centroids = F.normalize(text_centroids, p=2, dim=-1)
    
    W = images @ text_centroids.t()
    Q = descriptions @ text_centroids.t()
    
    alpha = torch.sigmoid(self.alpha)
    logits = alpha * W + (1 - alpha) * Q
    
    logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
    class_temps = torch.softmax(self.class_temps, dim=0)
    logits = logits * logit_scale * (1 + class_temps).unsqueeze(0)
    
    logits = self.dropout(logits)
    return logits
```

---

## INIT

```python
class CALOGModified(nn.Module):
    def __init__(self, num_classes=10, feature_dim=512, desc_dim=768):
        super().__init__()
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        self.desc_projection = nn.Sequential(
            nn.Linear(desc_dim, 1045),
            nn.GELU(),
            nn.Linear(1045, feature_dim)
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.6))
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.class_temps = nn.Parameter(torch.ones(num_classes))
        
        self.image_norm = nn.LayerNorm(feature_dim)
        self.desc_norm = nn.LayerNorm(feature_dim)
        self.text_norm = nn.LayerNorm(feature_dim)
        
        self.dropout = nn.Dropout(0.15)
```

