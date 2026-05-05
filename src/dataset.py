import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import BertTokenizer


class MedCLIPDataset(Dataset):
    def __init__(self, data_root, split, concept_file, image_size=224, max_length=128):
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

        with open(concept_file, "r") as f:
            self.concepts = json.load(f)

        self.samples = []
        split_dir = os.path.join(data_root, split)
        for study in os.listdir(split_dir):
            study_path = os.path.join(split_dir, study)
            if not os.path.isdir(study_path):
                continue
            for series in os.listdir(study_path):
                series_path = os.path.join(study_path, series)
                for img in os.listdir(series_path):
                    if img.endswith(".jpg"):
                        self.samples.append({
                            "path": os.path.join(series_path, img),
                            "study": study
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB").resize((self.image_size, self.image_size))
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0

        study_concepts = self.concepts.get(sample["study"], [])
        text = " . ".join(study_concepts) if study_concepts else "no finding"
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        all_concepts = self.concepts.get("all_concepts", [])
        concept_labels = torch.tensor([
            1.0 if c in study_concepts else 0.0 for c in all_concepts
        ], dtype=torch.float)

        return {
            "image": image,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "concept_labels": concept_labels,
        }
