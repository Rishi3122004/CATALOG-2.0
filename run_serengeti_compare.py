import glob
import os

from models import CATALOG_Base as base
from models import CATALOG_Base_modified as base_modified
from train.Base.Train_CATALOG_Base_out_domain import CATALOG_base
from train.Base.Train_CATALOG_Base_modified import CATALOG_base_modified
from utils import BaselineDataset, dataloader_baseline, build_optimizer


def latest_checkpoint(exp_name):
    pattern = os.path.join("Best", exp_name, "training_*", "best_model_params_*.pth")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def build_paths():
    base_dir = "features/Features_serengeti/standard_features"
    return {
        "train": os.path.join(base_dir, "Features_CATALOG_train_16.pt"),
        "val": os.path.join(base_dir, "Features_CATALOG_val_16.pt"),
        "test": os.path.join(base_dir, "Features_CATALOG_test_16.pt"),
        "text": os.path.join(base_dir, "Text_features_16.pt"),
    }


def train_original(paths):
    print("\n===== Training ORIGINAL CATALOG (Serengeti-only) =====")
    model = CATALOG_base(
        weight_Clip=0.6,
        num_epochs=8,
        batch_size=48,
        num_layers=1,
        dropout=0.27822,
        hidden_dim=1045,
        lr=0.07641,
        t=0.1,
        momentum=0.8409,
        patience=5,
        model=base,
        Dataset=BaselineDataset,
        Dataloader=dataloader_baseline,
        version="base",
        ruta_features_train=paths["train"],
        ruta_features_val=paths["val"],
        ruta_features_test1=paths["test"],
        ruta_features_test2=paths["val"],
        path_text_feat1=paths["text"],
        path_text_feat2=paths["text"],
        build_optimizer=build_optimizer,
        exp_name="exp_Base_SerengetiOnly",
        subset_size=10000,
    )

    model.train()

    ckpt = latest_checkpoint("exp_Base_SerengetiOnly")
    if ckpt is None:
        print("No checkpoint found for original model.")
        return None

    print(f"Original checkpoint: {ckpt}")
    print("\n----- Original Test Metrics (on Serengeti test split) -----")
    model.prueba_model(model_params_path=ckpt)
    return ckpt


def train_modified(paths):
    print("\n===== Training MODIFIED CATALOG (3 changes enabled) =====")
    print("Changes active: dynamic alpha + BERT/MLP bypass + modified loss")

    model = CATALOG_base_modified(
        weight_Clip=0.6,
        num_epochs=8,
        batch_size=48,
        num_layers=1,
        dropout=0.27822,
        hidden_dim=1045,
        lr=0.07641,
        t=0.1,
        momentum=0.8409,
        patience=5,
        model=base_modified,
        Dataset=BaselineDataset,
        Dataloader=dataloader_baseline,
        version="base_modified",
        ruta_features_train=paths["train"],
        ruta_features_val=paths["val"],
        ruta_features_test1=paths["test"],
        ruta_features_test2=paths["val"],
        path_text_feat1=paths["text"],
        path_text_feat2=paths["text"],
        build_optimizer=build_optimizer,
        exp_name="exp_Base_modified_SerengetiOnly",
        subset_size=10000,
    )

    ckpt = model.train()
    print(f"Modified checkpoint: {ckpt}")
    print("\n----- Modified Test Metrics (test + val-as-second-split) -----")
    model.prueba_model(model_params_path=ckpt)
    return ckpt


def main():
    paths = build_paths()

    for key, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required feature file for {key}: {path}")

    train_original(paths)
    train_modified(paths)


if __name__ == "__main__":
    main()
