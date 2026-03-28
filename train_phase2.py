#!/usr/bin/env python
"""
CATALOG Phase 2 Training Script
Enhanced multimodal model with:
- Fine-tunable BERT encoder
- Learnable fusion weights
- Attention-based fusion mechanism
- Hard negative mining
"""

import os
import sys
import torch
import time
import datetime
import pandas as pd
import numpy as np
import random
from sklearn.metrics import confusion_matrix, classification_report

# Add model path
sys.path.insert(0, 'models')

# Import Phase 2 model
from CATALOG_Base_Phase2 import LLaVA_CLIP_Phase2

# Dataset and utilities
from data.serengeti.split_images import SerengentiDataset
from utils import CATALOG_DataLoader

print("=" * 90)
print("  CATALOG PHASE 2 TRAINING")
print("  Enhanced Multimodal Learning with Advanced Fusion")
print("=" * 90)

class CATALOG_Phase2_Training:
    def __init__(self, weight_Clip, num_epochs, batch_size, num_layers, dropout, hidden_dim, 
                 lr, t, momentum, patience, model, Dataset, Dataloader, version, 
                 ruta_features_train, ruta_features_val, ruta_features_test, 
                 path_text_feat, build_optimizer, exp_name):
        self.ruta_features_train = ruta_features_train
        self.ruta_features_val = ruta_features_val
        self.ruta_features_test = ruta_features_test
        self.path_text_feat = path_text_feat
        self.weight_Clip = weight_Clip
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.t = t
        self.momentum = momentum
        self.patience = patience
        self.md = model
        self.dataset = Dataset
        self.dataloader = Dataloader
        self.version = version
        self.build_optimizer = build_optimizer
        self.exp_name = exp_name

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def train(self):
        self.set_seed(42)
        unique_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load text features
        text_features = torch.load(self.path_text_feat)
        text_features = text_features.float().to(device)

        # Create Phase 2 model
        projection_model = LLaVA_CLIP_Phase2(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            device=device,
            num_classes=10,
            enable_bert_tuning=True
        )
        projection_model = projection_model.float().to(device)
        
        print("\n[MODEL] Phase 2 Enhanced Architecture:")
        print(f"  - BERT Tuning: ENABLED")
        print(f"  - Learnable Fusion Weights: ENABLED")
        print(f"  - Attention Fusion: ENABLED")
        print(f"  - Hard Negative Mining: ENABLED")

        # DataLoaders
        dataloader = self.dataloader(self.ruta_features_train, self.batch_size, self.dataset)
        dataloader_val = self.dataloader(self.ruta_features_val, self.batch_size, self.dataset)
        dataloader_test = self.dataloader(self.ruta_features_test, self.batch_size, self.dataset)

        # Optimizer (note: now includes BERT and fusion parameters)
        optimizer, scheduler = self.build_optimizer(projection_model, 'sgd', self.lr, self.momentum, self.version)
        
        acc_best = 0
        counter = 0
        save_dir = f"Best/{self.exp_name}/training_{unique_id}/"
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n[TRAINING] Starting Phase 2 training...")
        print(f"  Save directory: {save_dir}")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Early stopping patience: {self.patience}\n")

        for epoch in range(self.num_epochs):
            print(epoch)
            
            # Training phase
            projection_model.train()
            time_in = time.time()
            running_loss = 0.0
            running_corrects = 0.0
            size = 0
            
            for batch in dataloader:
                image_features, description_embeddings, target_index = batch
                size += len(image_features)
                
                image_features = image_features.float().to(device)
                description_embeddings = description_embeddings.float().to(device)
                target_index = target_index.to(device)

                # Phase 2 forward with hard negative mining
                loss, acc, _ = projection_model(
                    description_embeddings, 
                    image_features, 
                    text_features, 
                    self.weight_Clip,
                    target_index,
                    self.t,
                    use_hard_mining=True  # Enable hard negative mining
                )

                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()

                running_loss += loss.item()
                running_corrects += float(acc)

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_corrects / size * 100

            # Validation phase
            projection_model.eval()
            running_loss_val = 0.0
            running_corrects_val = 0.0
            size_val = 0
            
            with torch.no_grad():
                for batch in dataloader_val:
                    image_features, description_embeddings, target_index = batch
                    size_val += len(image_features)
                    
                    image_features = image_features.float().to(device)
                    description_embeddings = description_embeddings.float().to(device)
                    target_index = target_index.to(device)

                    loss, acc, _ = projection_model.predict(
                        description_embeddings,
                        image_features,
                        text_features,
                        self.weight_Clip,
                        target_index,
                        self.t
                    )

                    running_loss_val += loss.item()
                    running_corrects_val += float(acc)

            val_loss = running_loss_val / len(dataloader_val)
            val_acc = running_corrects_val / size_val * 100

            time_for_epoch = time.time() - time_in

            print(f"Epoch [{epoch+1}/{self.num_epochs}]")
            print(f"Train loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}")
            print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
            print(f"Time for epoch [{time_for_epoch}]")

            # Save best model
            if val_acc > acc_best:
                acc_best = val_acc
                counter = 0
                model_params_path = os.path.join(save_dir, f"best_model_params_{self.num_layers}_{self.hidden_dim}.pth")
                torch.save(projection_model.state_dict(), model_params_path)
                print("Save model")
            else:
                print("The acc don't increase")
                counter += 1
                if counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Final test evaluation
        print("\n" + "=" * 90)
        print("  TEST SET EVALUATION")
        print("=" * 90)
        
        projection_model.eval()
        running_loss_test = 0.0
        running_corrects_test = 0.0
        size_test = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader_test:
                image_features, description_embeddings, target_index = batch
                size_test += len(image_features)
                
                image_features = image_features.float().to(device)
                description_embeddings = description_embeddings.float().to(device)
                target_index = target_index.to(device)

                loss, acc, preds = projection_model.predict(
                    description_embeddings,
                    image_features,
                    text_features,
                    self.weight_Clip,
                    target_index,
                    self.t
                )

                running_loss_test += loss.item()
                running_corrects_test += float(acc)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target_index.cpu().numpy())

        test_loss = running_loss_test / len(dataloader_test)
        test_acc = running_corrects_test / size_test * 100

        print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
        print("=" * 90)

        return {
            'best_val_acc': acc_best,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'model_path': model_params_path,
            'predictions': np.array(all_preds),
            'targets': np.array(all_targets)
        }

# Configuration
print("\n[CONFIG] Phase 2 Hyperparameters:")
config = {
    'weight_Clip': 0.60855,
    'num_epochs': 86,
    'batch_size': 26,
    'num_layers': 4,
    'dropout': 0.381,
    'hidden_dim': 1743,
    'lr': 0.0956,  # Slightly higher for BERT fine-tuning
    't': 0.1,
    'momentum': 0.8162,
    'patience': 20,
    'exp_name': 'exp_Base_In_domain_Serengeti_Phase2'
}

for key, val in config.items():
    print(f"  {key}: {val}")

print("\n[STARTING] Phase 2 training with enhanced architecture...")
print("  Starting time:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
