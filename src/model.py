import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import BertModel


class MedCLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.image_encoder == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.image_encoder = nn.Sequential(*list(backbone.children())[:-1])
            image_dim = 2048
        elif config.image_encoder == "vit_base_patch16":
            from transformers import ViTModel
            self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
            image_dim = 768
        else:
            raise ValueError(f"Unsupported image encoder: {config.image_encoder}")

        self.text_encoder = BertModel.from_pretrained(config.text_encoder)
        text_dim = 768

        self.image_proj = nn.Linear(image_dim, config.embed_dim)
        self.text_proj = nn.Linear(text_dim, config.embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def encode_image(self, images):
        if self.config.image_encoder == "resnet50":
            feats = self.image_encoder(images).flatten(1)
        else:
            feats = self.image_encoder(images).pooler_output
        return F.normalize(self.image_proj(feats), dim=-1)

    def encode_text(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        feats = outputs.pooler_output
        return F.normalize(self.text_proj(feats), dim=-1)

    def forward(self, images, input_ids, attention_mask):
        return self.encode_image(images), self.encode_text(input_ids, attention_mask)
