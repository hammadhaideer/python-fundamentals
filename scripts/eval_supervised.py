import os
import json
import yaml
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import roc_auc_score

from src.model import MedCLIP
from src.dataset import MedCLIPDataset


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["chexpert", "mimic"], required=True)
    parser.add_argument("--label_frac", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MedCLIP(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    data_root = os.environ.get("MIMIC_CXR_ROOT") if args.dataset == "mimic" else os.environ.get("CHEXPERT_ROOT")
    full_dataset = MedCLIPDataset(
        data_root=data_root,
        split="train",
        concept_file=config["concept_file"],
        image_size=config["image_size"],
        max_length=config["max_text_length"],
    )

    num_samples = int(len(full_dataset) * args.label_frac)
    indices = np.random.choice(len(full_dataset), num_samples, replace=False)
    subset = torch.utils.data.Subset(full_dataset, indices)
    loader = DataLoader(subset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])

    probe = LinearProbe(config["embed_dim"], len(full_dataset.concepts.get("all_concepts", []))).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        probe.train()
        total_loss = 0
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["concept_labels"].to(device)

            with torch.no_grad():
                img_feats, _ = model(images, None, None)
            logits = probe(img_feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(loader):.4f}")

    probe.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["concept_labels"]
            img_feats, _ = model(images, None, None)
            logits = probe(img_feats)
            preds = torch.sigmoid(logits)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    auc = roc_auc_score(all_labels.ravel(), all_preds.ravel())

    results = {
        "dataset": args.dataset,
        "label_frac": args.label_frac,
        "auc": float(auc),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Supervised AUROC (linear probe): {auc:.4f}")


if __name__ == "__main__":
    main()
