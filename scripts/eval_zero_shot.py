import os
import json
import yaml
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.model import MedCLIP
from src.dataset import MedCLIPDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["chexpert", "mimic"], required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedCLIP(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    data_root = os.environ.get("MIMIC_CXR_ROOT") if args.dataset == "mimic" else os.environ.get("CHEXPERT_ROOT")
    dataset = MedCLIPDataset(
        data_root=data_root,
        split=args.split,
        concept_file=config["concept_file"],
        image_size=config["image_size"],
        max_length=config["max_text_length"],
    )
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            concept_labels = batch["concept_labels"].cpu().numpy()

            img_feats, txt_feats = model(images, input_ids, attention_mask)
            sim = (img_feats @ txt_feats.T).cpu().numpy()

            scores = sim.max(axis=1)
            all_scores.extend(scores)
            all_labels.extend(concept_labels.max(axis=1))

    auc = roc_auc_score(all_labels, all_scores)
    results = {"dataset": args.dataset, "split": args.split, "auc": float(auc)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Zero-shot AUROC: {auc:.4f}")


if __name__ == "__main__":
    main()
