import os
import torch
import time
import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import random

class CATALOG_base_modified:
    def __init__(self, weight_Clip, num_epochs, batch_size, num_layers, dropout, hidden_dim, lr, t, momentum, patience, model, Dataset, Dataloader, version, ruta_features_train, ruta_features_val, ruta_features_test1, ruta_features_test2, path_text_feat1, path_text_feat2, build_optimizer, exp_name, subset_size=None):
        self.ruta_features_train = ruta_features_train
        self.ruta_features_val = ruta_features_val
        self.ruta_features_test1 = ruta_features_test1
        self.ruta_features_test2 = ruta_features_test2
        self.path_text_feat1 = path_text_feat1
        self.path_text_feat2 = path_text_feat2
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
        self.subset_size = subset_size

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

        text_features = torch.load(self.path_text_feat1)
        text_features = text_features.to(device)

        text_features2 = torch.load(self.path_text_feat2)
        text_features2 = text_features2.to(device)

        projection_model = self.md.LLaVA_CLIP_modified(hidden_dim=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout, device=device)
        projection_model = projection_model.to(device)

        # Get your DataLoader with subset
        dataloader = self.dataloader(self.ruta_features_train, self.batch_size, self.dataset, subset_size=self.subset_size)
        dataloader_val = self.dataloader(self.ruta_features_val, self.batch_size, self.dataset)
        dataloader_cis_test = self.dataloader(self.ruta_features_test1, self.batch_size, self.dataset)
        dataloader_trans_test = self.dataloader(self.ruta_features_test2, self.batch_size, self.dataset)

        # Configurate optimizer for training
        optimizer, scheduler = self.build_optimizer(projection_model, 'sgd', self.lr, self.momentum, self.version)
        acc_best = 0
        counter = 0
        for epoch in range(self.num_epochs):
            print(epoch)
            # Training
            projection_model.train()
            time_in = time.time()
            running_loss = 0.0
            running_corrects = 0.0
            size = 0
            for batch in dataloader:
                image_features, _, target_index = batch  # Bypass: ignore description_embeddings
                size += len(image_features)
                image_features = image_features.to(device)
                target_index = target_index.to(device)

                loss, acc, _ = projection_model(image_features, text_features, target_index, self.t)

                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()

                running_loss += loss.item()
                running_corrects += float(acc)

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = (running_corrects / size) * 100

            # validation
            projection_model.eval()
            running_loss_val = 0
            running_corrects_val = 0.0
            size_val = 0

            with torch.no_grad():
                for batch_val in dataloader_val:
                    image_features_val, _, target_index_val = batch_val
                    size_val += len(image_features_val)
                    image_features_val = image_features_val.to(device)
                    target_index_val = target_index_val.to(device)

                    loss_val, acc_val, _ = projection_model(image_features_val, text_features, target_index_val, self.t)

                    running_loss_val += loss_val.item()
                    running_corrects_val += float(acc_val)

            epoch_loss_val = running_loss_val / len(dataloader_val)
            epoch_acc_val = (running_corrects_val / size_val) * 100

            time_end = time.time()
            total_time = time_end - time_in
            if epoch % 1 == 0:
                print(f"Epoch [{epoch + 1}/{self.num_epochs}]")
                print('Train loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, epoch_acc))
                print('Val loss: {:.4f}, Val acc: {:.4f}'.format(epoch_loss_val, epoch_acc_val))
                print(f"Time for epoch [{total_time}]")
                if epoch_acc_val > acc_best:
                    print('Save model')
                    acc_best = epoch_acc_val
                    counter = 0
                    os.makedirs(f'Best/{self.exp_name}/training_{unique_id}', exist_ok=True)
                    model_params_path = f'Best/{self.exp_name}/training_{unique_id}/best_model_params_{self.num_layers}_{self.hidden_dim}.pth'
                    torch.save(projection_model.state_dict(), model_params_path)
                else:
                    counter += 1
                    if counter >= self.patience:
                        print("Early stopping")
                        break

        return model_params_path

    def prueba_model(self, model_params_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        text_features2 = torch.load(self.path_text_feat2)
        text_features2 = text_features2.to(device)

        projection_model = self.md.LLaVA_CLIP_modified(hidden_dim=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout, device=device)
        projection_model.load_state_dict(torch.load(model_params_path))
        projection_model = projection_model.to(device)
        projection_model.eval()

        dataloader_cis_test = self.dataloader(self.ruta_features_test1, self.batch_size, self.dataset)
        dataloader_trans_test = self.dataloader(self.ruta_features_test2, self.batch_size, self.dataset)

        # Cis test
        running_corrects_cis = 0.0
        size_cis = 0
        with torch.no_grad():
            for batch in dataloader_cis_test:
                image_features, _, target_index = batch
                size_cis += len(image_features)
                image_features = image_features.to(device)
                target_index = target_index.to(device)
                _, acc, _ = projection_model(image_features, text_features2, target_index, self.t)
                running_corrects_cis += float(acc)
        acc_cis = (running_corrects_cis / size_cis) * 100

        # Trans test
        running_corrects_trans = 0.0
        size_trans = 0
        with torch.no_grad():
            for batch in dataloader_trans_test:
                image_features, _, target_index = batch
                size_trans += len(image_features)
                image_features = image_features.to(device)
                target_index = target_index.to(device)
                _, acc, _ = projection_model(image_features, text_features2, target_index, self.t)
                running_corrects_trans += float(acc)
        acc_trans = (running_corrects_trans / size_trans) * 100

        print(f'Cis Test Acc: {acc_cis:.2f}%')
        print(f'Trans Test Acc: {acc_trans:.2f}%')

    def prueba_model_top_3(self, model_params_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        text_features2 = torch.load(self.path_text_feat2)
        text_features2 = text_features2.to(device)

        projection_model = self.md.LLaVA_CLIP_modified(hidden_dim=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout, device=device)
        projection_model.load_state_dict(torch.load(model_params_path))
        projection_model = projection_model.to(device)
        projection_model.eval()

        dataloader_cis_test = self.dataloader(self.ruta_features_test1, self.batch_size, self.dataset)
        dataloader_trans_test = self.dataloader(self.ruta_features_test2, self.batch_size, self.dataset)

        # Cis top3
        running_corrects_cis = 0.0
        size_cis = 0
        with torch.no_grad():
            for batch in dataloader_cis_test:
                image_features, _, target_index = batch
                size_cis += len(image_features)
                image_features = image_features.to(device)
                target_index = target_index.to(device)
                pred = projection_model.predict_top_3(image_features, text_features2, target_index, self.t)
                pred = pred.cpu()
                correct = pred.eq(target_index.view(1, -1).expand_as(pred))
                correct_k = float(correct[:3].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
                running_corrects_cis += correct_k
        acc_cis = (running_corrects_cis / size_cis) * 100

        # Trans top3
        running_corrects_trans = 0.0
        size_trans = 0
        with torch.no_grad():
            for batch in dataloader_trans_test:
                image_features, _, target_index = batch
                size_trans += len(image_features)
                image_features = image_features.to(device)
                target_index = target_index.to(device)
                pred = projection_model.predict_top_3(image_features, text_features2, target_index, self.t)
                pred = pred.cpu()
                correct = pred.eq(target_index.view(1, -1).expand_as(pred))
                correct_k = float(correct[:3].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
                running_corrects_trans += correct_k
        acc_trans = (running_corrects_trans / size_trans) * 100

        print(f'Cis Test Top3 Acc: {acc_cis:.2f}%')
        print(f'Trans Test Top3 Acc: {acc_trans:.2f}%')