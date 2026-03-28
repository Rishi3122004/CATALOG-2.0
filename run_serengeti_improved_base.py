import glob
import os

from models import CATALOG_Base as base
from train.Base.Train_CATALOG_Base_out_domain import CATALOG_base
from utils import BaselineDataset, dataloader_baseline, build_optimizer


def latest_checkpoint(exp_name):
    pattern = os.path.join("Best", exp_name, "training_*", "best_model_params_*.pth")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def main():
    paths = {
        "train": "features/Features_serengeti/standard_features/Features_CATALOG_train_16.pt",
        "val": "features/Features_serengeti/standard_features/Features_CATALOG_val_16.pt",
        "test": "features/Features_serengeti/standard_features/Features_CATALOG_test_16.pt",
        "text": "features/Features_serengeti/standard_features/Text_features_16.pt",
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {name} file: {path}")

    model = CATALOG_base(
        weight_Clip=0.6,
        num_epochs=20,
        batch_size=48,
        num_layers=1,
        dropout=0.27822,
        hidden_dim=1045,
        lr=0.03,
        t=0.1,
        momentum=0.9,
        patience=7,
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
        exp_name="exp_Base_Serengeti_Improved",
        subset_size=None,
    )

    print("===== Training improved base on Serengeti =====")
    model.train()

    ckpt = latest_checkpoint("exp_Base_Serengeti_Improved")
    if ckpt is None:
        raise RuntimeError("No improved checkpoint found after training.")

    print(f"Improved checkpoint: {ckpt}")
    print("===== Evaluating improved base on Serengeti test =====")
    model.prueba_model(model_params_path=ckpt)


if __name__ == "__main__":
    main()
