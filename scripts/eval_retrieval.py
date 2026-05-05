import os
import json
import yaml
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader

from src.model import MedCLIP
from src.dataset import MedCLIPDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["chexpert", "mimic"], required=True)
    parser.add_argument("--k", type=int, default=10)
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
        split="test",
        concept_file=config["concept_file"],
        image_size=config["image_size"],
        max_length=config["max_text_length"],
    )
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    all_img_feats = []
    all_txt_feats = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            img_feats, txt_feats = model(images, input_ids, attention_mask)
            all_img_feats.append(img_feats.cpu().numpy())
            all_txt_feats.append(txt_feats.cpu().numpy())

    img_feats = np.concatenate(all_img_feats, axis=0)
    txt_feats = np.concatenate(all_txt_feats, axis=0)
    sim = img_feats @ txt_feats.T

    recalls = []
    for k in [1, 5, 10]:
        if args.k == k:
            ranks = np.argsort(-sim, axis=1)[:, :k]
            recall = np.mean([1 if i in ranks[i] else 0 for i in range(len(ranks))])
            recalls.append(recall)

    results = {
        "dataset": args.dataset,
        "recall@1": recalls[0] if 1 in [1,5,10] else None,
        "recall@5": recalls[1] if 5 in [1,5,10] else None,
        "recall@10": recalls[2] if 10 in [1,5,10] else None,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Recall@{args.k}: {recalls[args.k//5 if args.k>1 else 0]:.4f}")


if __name__ == "__main__":
    main()
