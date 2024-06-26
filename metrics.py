import torch
import torcheval.metrics.functional as F

from typing import Tuple, Literal
import logging
logging.getLogger().setLevel(logging.ERROR)

class AspectF1Score:
    def __init__(
        self,
        aspect_thresholds: float | torch.Tensor = 0.5,
        reduction: Literal["mean", "none"] = "mean"
    ) -> None:
        if isinstance(aspect_thresholds, float):
            aspect_thresholds = torch.full((11,), aspect_thresholds)
        self.aspect_thresholds = aspect_thresholds
        self.reduction = reduction
    
    def __call__(
        self, 
        y_hat: Tuple[torch.Tensor, torch.Tensor],
        y: Tuple[torch.Tensor, torch.Tensor]
    ):
        a_hat, p_hat = y_hat
        a, p = y

        a_hat = a_hat.sigmoid()

        scores = []
        for i_a in torch.arange(11):
            input = a_hat[..., i_a]
            target = a[..., i_a]

            score = F.binary_f1_score(input, target, threshold=self.aspect_thresholds[i_a])
            scores.append(score)
        
        result = torch.tensor(scores)
        if self.reduction == "mean":
            result = result.mean()
        return result
    
class PolarityF1Score:
    def __init__(
        self,
        average: Literal["weighted", "macro", "micro"] = "macro",
        reduction: Literal["mean", "none"] = "mean"
    ) -> None:
        self.average = average
        self.reduction = reduction
    
    def __call__(
        self, 
        y_hat: Tuple[torch.Tensor, torch.Tensor],
        y: Tuple[torch.Tensor, torch.Tensor]
    ):
        a_hat, p_hat = y_hat
        a, p = y

        scores = []
        for i_a in torch.arange(10):
            input = p_hat[..., i_a, :]
            target = p[..., i_a]

            input = input[target != -1]
            target = target[target != -1]

            score = F.multiclass_f1_score(input, target, num_classes=3, average=self.average)
            scores.append(score)

        result = torch.tensor(scores)
        if self.reduction == "mean":
            result = result.nanmean()
        return result