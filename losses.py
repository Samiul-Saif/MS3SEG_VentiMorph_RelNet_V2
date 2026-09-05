"""
Deliberately small loss surface: weighted cross-entropy + soft Dice, applied
identically at the main head and at every deep-supervision head. Everything
that made the previous custom model hard to optimize (9 simultaneously
weighted terms: CE, Dice, two focal-Tversky terms, a boundary L1 term, a
ventricle auxiliary BCE+Dice term, a normal/abnormal "confusion" penalty, a
contrastive prototype loss, and an evidential Dirichlet-KL uncertainty loss,
several of them ramped in on hand-picked schedules) is removed on purpose.

One optional extra term is kept: `lesion_tversky_weight`, a single
recall-leaning Tversky term on the abnormal-WMH class only. It defaults to
0.0 (off). Turn it on as ONE controlled ablation, not by default, so any
reported gain is attributable to a single, clearly isolated change.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 4, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1)[:, 1:]
    one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()[:, 1:]
    intersection = (prob * one_hot).sum((0, 2, 3))
    denom = prob.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
    return 1.0 - ((2.0 * intersection + eps) / (denom + eps)).mean()


def focal_tversky(logits: torch.Tensor, target: torch.Tensor, class_id: int,
                   alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1)[:, class_id]
    truth = (target == class_id).float()
    tp = (prob * truth).sum()
    fp = (prob * (1 - truth)).sum()
    fn = ((1 - prob) * truth).sum()
    tversky = (tp + eps) / (tp + alpha * fn + beta * fp + eps)
    return (1.0 - tversky) ** gamma


class DiceCELoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, lesion_tversky_weight: float = 0.0, num_classes: int = 4):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.lesion_tversky_weight = lesion_tversky_weight
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.class_weights)
        dice = soft_dice_loss(logits, target, self.num_classes)
        total = ce + dice
        if self.lesion_tversky_weight > 0.0:
            total = total + self.lesion_tversky_weight * focal_tversky(logits, target, class_id=3)
        return total

    def with_deep_supervision(self, main_logits, aux_logits_list, target,
                               aux_weights=(0.5, 0.25, 0.125)) -> torch.Tensor:
        total = self.forward(main_logits, target)
        for aux, w in zip(aux_logits_list, aux_weights):
            aux_up = F.interpolate(aux, size=target.shape[-2:], mode="bilinear", align_corners=False)
            total = total + w * self.forward(aux_up, target)
        return total


def binary_dice_ce_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Stage-1 ventricle loss: plain BCE-with-logits + Dice on a single channel."""
    target_f = target.float()
    bce = F.binary_cross_entropy_with_logits(logits.squeeze(1), target_f)
    prob = torch.sigmoid(logits.squeeze(1))
    dice = 1.0 - (2.0 * (prob * target_f).sum() + eps) / (prob.sum() + target_f.sum() + eps)
    return bce + dice
