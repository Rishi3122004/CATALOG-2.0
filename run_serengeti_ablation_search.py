import json
import os

from models import CATALOG_Base as base
from train.Base.Train_CATALOG_Base_out_domain import CATALOG_base
from utils import BaselineDataset, dataloader_baseline, build_optimizer


def build_paths():
    root = "features/Features_serengeti/standard_features"
    return {
        "train": os.path.join(root, "Features_CATALOG_train_16.pt"),
        "val": os.path.join(root, "Features_CATALOG_val_16.pt"),
        "test": os.path.join(root, "Features_CATALOG_test_16.pt"),
        "text": os.path.join(root, "Text_features_16.pt"),
    }


def run_trial(name, weight_clip, lr, fusion_enabled, fusion_init, epochs=20, patience=7):
    paths = build_paths()
    model = CATALOG_base(
        weight_Clip=weight_clip,
        num_epochs=epochs,
        batch_size=48,
        num_layers=1,
        dropout=0.27822,
        hidden_dim=1045,
        lr=lr,
        t=0.1,
        momentum=0.9,
        patience=patience,
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
        exp_name=f"exp_Ablation_{name}",
        subset_size=None,
        model_kwargs={
            "enable_classifier_fusion": fusion_enabled,
            "fusion_init": fusion_init,
        },
    )

    train_info = model.train()
    eval_info = model.prueba_model(train_info["model_params_path"])

    result = {
        "name": name,
        "weight_clip": weight_clip,
        "lr": lr,
        "fusion_enabled": fusion_enabled,
        "fusion_init": fusion_init,
        "best_val_acc": train_info["best_val_acc"],
        "final_train_loop_cis_test_acc": train_info["final_cis_test_acc"],
        "eval_cis_test_acc": eval_info["cis_test_acc"],
        "model_params_path": train_info["model_params_path"],
    }
    return result


def main():
    trials = [
        ("baseline_like", 0.6, 0.03, False, -2.2),
        ("fusion_default", 0.6, 0.03, True, -2.2),
        ("fusion_more_clip", 0.75, 0.03, True, -2.2),
        ("fusion_more_bert", 0.45, 0.03, True, -2.2),
        ("fusion_low_lr", 0.6, 0.01, True, -2.2),
    ]

    results = []
    for trial in trials:
        name, w, lr, fusion_enabled, fusion_init = trial
        print(f"\\n===== Running trial: {name} =====")
        result = run_trial(name, w, lr, fusion_enabled, fusion_init)
        print(f"Trial {name} eval_cis_test_acc: {result['eval_cis_test_acc']:.4f}")
        results.append(result)

    results.sort(key=lambda r: r["eval_cis_test_acc"], reverse=True)

    os.makedirs("Best/ablation_reports", exist_ok=True)
    report_path = "Best/ablation_reports/serengeti_ablation_results.json"
    with open(report_path, "w", encoding="ascii") as f:
        json.dump(results, f, indent=2)

    print("\\n===== Top trials =====")
    for i, r in enumerate(results[:3], 1):
        print(f"{i}. {r['name']} -> {r['eval_cis_test_acc']:.4f}% | ckpt={r['model_params_path']}")

    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
