import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticMatchingLoss(nn.Module):
    def __init__(self, margin=0.2, hard_negative_mining=True):
        super().__init__()
        self.margin = margin
        self.hard_negative_mining = hard_negative_mining

    def forward(self, image_features, text_features, concept_labels, threshold=0.5):
        batch_size = image_features.shape[0]
        sim = image_features @ text_features.T
        pos_mask = (concept_labels @ concept_labels.T) > threshold
        pos_mask.fill_diagonal_(False)

        loss = 0.0
        num_pos = 0

        for i in range(batch_size):
            pos_indices = torch.where(pos_mask[i])[0]
            if len(pos_indices) == 0:
                continue

            pos_sim = sim[i, pos_indices]

            if self.hard_negative_mining:
                neg_mask = ~pos_mask[i]
                neg_mask[i] = False
                if neg_mask.any():
                    hard_neg_sim = sim[i, neg_mask].max()
                else:
                    continue
            else:
                neg_mask = ~pos_mask[i]
                neg_mask[i] = False
                if neg_mask.any():
                    hard_neg_sim = sim[i, neg_mask].mean()
                else:
                    continue

            loss += F.relu(self.margin - pos_sim + hard_neg_sim).sum()
            num_pos += len(pos_indices)

        if num_pos > 0:
            loss = loss / num_pos
        else:
            loss = torch.tensor(0.0, device=image_features.device)

        return loss
