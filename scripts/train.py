import os
import yaml
import torch
import argparse
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.model import MedCLIP
from src.dataset import MedCLIPDataset
from src.losses import SemanticMatchingLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedCLIP(config).to(device)
    loss_fn = SemanticMatchingLoss(margin=config["margin"])
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scaler = GradScaler() if config["use_amp"] else None

    train_dataset = MedCLIPDataset(
        data_root=os.environ["MIMIC_CXR_ROOT"],
        split="train",
        concept_file=config["concept_file"],
        image_size=config["image_size"],
        max_length=config["max_text_length"],
    )
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch in pbar:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            concept_labels = batch["concept_labels"].to(device)

            optimizer.zero_grad()
            if scaler:
                with autocast():
                    img_feats, txt_feats = model(images, input_ids, attention_mask)
                    loss = loss_fn(img_feats, txt_feats, concept_labels, threshold=config["concept_threshold"])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                img_feats, txt_feats = model(images, input_ids, attention_mask)
                loss = loss_fn(img_feats, txt_feats, concept_labels, threshold=config["concept_threshold"])
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        print(f"Epoch {epoch+1} average loss: {total_loss / len(train_loader):.4f}")

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/medclip_epoch{epoch+1}.pth")


if __name__ == "__main__":
    main()
