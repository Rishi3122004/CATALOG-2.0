import clip
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.optim as optim

#Datasets and Dataloaders
class BaselineDataset(Dataset):
    def __init__(self,json_path, subset_size=None):
        self.root_dir = json_path
        self.subset_size = subset_size
        self.samples = self._load_samples()


    def _load_samples(self):
        samples=[]
        data_dict=torch.load(self.root_dir)
        
        # Handle both old dict format and new stacked tensor format
        if isinstance(data_dict, dict) and 'image_features' in data_dict:
            # New stacked tensor format
            image_features_all = data_dict['image_features'].float()
            desc_embeddings_all = data_dict['description_embeddings'].float()
            target_indices_all = data_dict['target_index']
            
            num_samples = len(target_indices_all)
            if self.subset_size is not None:
                num_samples = min(self.subset_size, num_samples)
            
            for i in range(num_samples):
                img_feat = image_features_all[i]
                desc_emb = desc_embeddings_all[i]
                target_idx = target_indices_all[i]
                samples.append([img_feat, desc_emb, target_idx])
        else:
            # Old dict format with image names as keys
            keys = list(data_dict.keys())
            if self.subset_size is not None:
                keys = keys[:self.subset_size]
            for key in keys:
                desc_emb = data_dict[key]['description_embeddings']
                if desc_emb is None:
                    desc_emb = torch.zeros(768)
                else:
                    desc_emb = desc_emb[0]
                img_feat = data_dict[key]['image_features'][0].float()
                desc_emb = desc_emb.float()
                samples.append([img_feat, desc_emb, data_dict[key]['target_index']])
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_features, description_embeddings, target_index= self.samples[idx]
        return image_features, description_embeddings, target_index

def dataloader_baseline(root_dir, batch_size,BaselineDataset, subset_size=None):
    dataset = BaselineDataset(root_dir, subset_size=subset_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)






#Tuning version
class TuningDataset(Dataset):
    def __init__(self,json_path, subset_size=None):
        self.root_dir = json_path
        self.subset_size = subset_size
        self.samples = self._load_samples()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        _, self.preprocess_clip = clip.load('ViT-B/16', self.device)


    def _load_samples(self):
        samples=[]
        data_dict=torch.load(self.root_dir)
        keys = list(data_dict.keys())
        if self.subset_size is not None:
            keys = keys[:self.subset_size]
        for key in keys:
            samples.append([data_dict[key]['image_features'],data_dict[key]['description_embeddings'][0],data_dict[key]['target_index']])
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_features, description_embeddings, target_index = self.samples[idx]
        image_features = self.preprocess_clip(Image.open(image_features).convert("RGB")).unsqueeze(0)  # .to(self.device)
        image_features = image_features[0]
        return image_features, description_embeddings, target_index

def dataloader_Tuning(root_dir, batch_size,TuningDataset, subset_size=None):
    dataset = TuningDataset(root_dir, subset_size=subset_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def build_optimizer( projection_model, optimizer, learning_rate, momentum, version):
    if hasattr(projection_model, 'description_encoder'):
        params1 = {"params": projection_model.description_encoder.parameters(), "lr": learning_rate,
                   "momentum": momentum}
    else:
        params1 = []
    params2 = {"params": projection_model.logit_scale_CLIP, "lr": learning_rate, "momentum": momentum}
    if hasattr(projection_model, 'logit_scale_LLaVA'):
        params3 = {"params": projection_model.logit_scale_LLaVA, "lr": learning_rate, "momentum": momentum}
    else:
        params3 = []

    if hasattr(projection_model, 'image_classifier'):
        params_img_cls = {"params": projection_model.image_classifier.parameters(), "lr": learning_rate,
                          "momentum": momentum}
    else:
        params_img_cls = []

    if hasattr(projection_model, 'fusion_logit'):
        params_fusion = {"params": projection_model.fusion_logit, "lr": learning_rate, "momentum": momentum}
    else:
        params_fusion = []

    scheduler = None  # Inicializa el scheduler como None


    if optimizer == "sgd":
        if version == 'base':
            optimizer = optim.SGD([params1, params2, params3, params_img_cls, params_fusion], lr=learning_rate, momentum=momentum)
        elif version == 'base_modified':
            if hasattr(projection_model, 'alpha_param'):
                params4 = {"params": projection_model.alpha_param, "lr": learning_rate, "momentum": momentum}
            else:
                params4 = []
            optimizer = optim.SGD([params2, params4], lr=learning_rate, momentum=momentum)
        elif version == 'projection':
            params4 = {"params": projection_model.proyection_Img_CLIP.parameters(), "lr": learning_rate,
                       "momentum": momentum}
            params5 = {"params": projection_model.proyection_txt_CLIP.parameters(), "lr": learning_rate,
                       "momentum": momentum}
            optimizer = optim.SGD([params1, params2, params3, params4, params5], lr=learning_rate,
                                  momentum=momentum)
        elif version == 'fine_tuning':
            params6 = {"params": projection_model.model_clip.visual.parameters(), "lr": learning_rate,
                       "momentum": momentum}
            optimizer = optim.SGD([params1, params2, params3, params6], lr=learning_rate,momentum=momentum)#, weight_decay=0.2
            T_max = 50
            eta_min = learning_rate
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

        elif version == 'fine_tuning_last_layer':
            params7 = {"params": projection_model.model_clip.visual.proj, "lr": learning_rate, "momentum": momentum}
            optimizer = optim.SGD([params1, params2, params3, params7], lr=learning_rate,momentum=momentum, weight_decay=0.2)
            T_max = 50
            eta_min = learning_rate
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

        elif version == 'base_modified':
            optimizer = optim.SGD([{"params": projection_model.logit_scale_CLIP, "lr": learning_rate, "momentum": momentum},
                                   {"params": projection_model.alpha_param, "lr": learning_rate, "momentum": momentum}], lr=learning_rate, momentum=momentum)

    return optimizer,scheduler


