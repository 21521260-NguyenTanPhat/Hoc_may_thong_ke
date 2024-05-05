import torch
from torch import nn
import torcheval.metrics.functional as F

from typing import Tuple, Literal
import logging
logging.getLogger().setLevel(logging.ERROR)

def stl_aspect_f1(
    y_hat: Tuple[torch.Tensor, torch.Tensor],
    y: Tuple[torch.Tensor, torch.Tensor],
    OTHERS_threshold: float = 0.5
):
    a_hat, o_hat = y_hat
    a, o = y

    scores = []
    # Aspect F1
    for i_a in torch.arange(10):
        input = (a_hat.select(-2, i_a).argmax(-1) > 0).int()
        target = (a[..., i_a] > 0).flatten().int()        

        score = F.binary_f1_score(input, target)
        scores.append(score)
    
    # OTHERS F1
    OTHERS_f1 = F.binary_f1_score(o_hat.flatten(), o.flatten(), threshold=OTHERS_threshold)
    scores.append(OTHERS_f1)

    return torch.tensor(scores).mean()


def mtl_aspect_f1(
    y_hat: Tuple[torch.Tensor, torch.Tensor],
    y: Tuple[torch.Tensor, torch.Tensor],
    aspect_threshold: float = 0.5
):
    a_hat, p_hat = y_hat
    a, p = y

    a_hat = a_hat.sigmoid()

    scores = []
    for i_a in torch.arange(11):
        input = a_hat[..., i_a]
        target = a[..., i_a]

        score = F.binary_f1_score(input, target, threshold=aspect_threshold)
        scores.append(score)
    
    return torch.tensor(scores).mean()


def stl_polarity_f1(
    y_hat: Tuple[torch.Tensor, torch.Tensor],
    y: Tuple[torch.Tensor, torch.Tensor],
):
    a_hat, o_hat = y_hat
    a, o = y




class AspectF1Score(nn.Module):
    def __init__(
        self,
        task_type: Literal["stl", "mtl"] = "stl",
        *args, **kwargs
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.args = args
        self.kwargs = kwargs

        if task_type == "stl":
            self.eval_fn = stl_aspect_f1
        elif task_type == "mtl":
            self.eval_fn = mtl_aspect_f1
    
    def forward(
        self,
        y_hat: Tuple[torch.Tensor, torch.Tensor],
        y: Tuple[torch.Tensor, torch.Tensor],
    ):
        with torch.no_grad():
            return self.eval_fn(y_hat, y, *self.args, **self.kwargs)