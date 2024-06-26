import torch
import torch.nn as nn

from typing import Tuple

class Loss(nn.Module):
    def __init__(
        self,
        a_label_weight: torch.Tensor = 1,
        aspect_weight: float = 1,
        polarity_weight: float = 1
    ):
        super(Loss, self).__init__()
        self.a_label_weight = a_label_weight
        self.w_a = aspect_weight
        self.w_p = polarity_weight

        self.aspect_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.polarity_fn = nn.CrossEntropyLoss(reduction="none")
    
    def compute_a_loss(
        self,
        y_hat: Tuple[torch.Tensor, torch.Tensor],
        y: Tuple[torch.Tensor, torch.Tensor]
    ):
        a_hat, p_hat = y_hat
        a, p = y

        a_loss = self.aspect_fn(a_hat, a)
        a_loss = torch.mean(a_loss * self.a_label_weight)
        return a_loss

    def compute_p_loss(
        self,
        y_hat: Tuple[torch.Tensor, torch.Tensor],
        y: Tuple[torch.Tensor, torch.Tensor]
    ):
        a_hat, p_hat = y_hat # Nx(A+1), NxAxP
        a, p = y # Nx(A+1), NxA

        a = a[..., :-1]
        p[p == -1] = 0
        p_loss = self.polarity_fn(p_hat.transpose(-1, -2), p) * a
        p_loss = p_loss.mean()
        return p_loss
    
    def forward(
        self,
        y_hat: Tuple[torch.Tensor, torch.Tensor],
        y: Tuple[torch.Tensor, torch.Tensor]
    ):
        a_hat, p_hat = y_hat # Nx(A+1), NxAxP
        a, p = y # Nx(A+1), NxA

        a_loss = self.compute_a_loss(y_hat, y)
        p_loss = self.compute_p_loss(y_hat, y)

        loss = self.w_a * a_loss + self.w_p * p_loss
        return loss