import torch
from torch import nn

from typing import Tuple, Optional, Literal, Any


class MultiLabelFocalLoss(nn.Module):
    def __init__(
        self, 
        alpha: float | torch.FloatTensor = 1,
        gamma: float = 2,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.gamma = gamma

        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor
    ):
        a = self.alpha
        g = self.gamma
        bce = self.bce
        loss = torch.sum(
            a*(1-y_hat.sigmoid())**g*bce(y_hat, y)
        )
        return loss.mean()


class STL_ViSFDLoss(nn.Module):
    def __init__(
        self, 
        aspect_weight: float = 1.,
        OTHERS_weight: float = 1.,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.w_a = aspect_weight
        self.w_OTHERS = OTHERS_weight

        self.aspect_fn = nn.CrossEntropyLoss()
        self.OTHERS_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, 
        y_hat: Tuple[torch.Tensor, torch.Tensor], 
        y: Tuple[torch.Tensor, torch.Tensor]
    ):
        a_hat, o_hat = y_hat # NxAxP, Nx1
        a, o = y # NxA, ...

        if a.ndim == 1:
            a = a.unsqueeze(0)

        a_loss = self.aspect_fn(a_hat.transpose(-1, -2), a)
        OTHERS_loss = self.OTHERS_fn(o_hat, o.reshape_as(o_hat))
        loss = self.w_a * a_loss +  self.w_OTHERS * OTHERS_loss
        return loss


class MTL_ViSFDLoss(nn.Module):
    def __init__(
        self, 
        aspect_alpha: float = 0.25,
        aspect_gamma: float = 2,
        aspect_weight: float = 1.,
        polarity_weight: float = 1.,
        OTHERS_index: Optional[int] = None,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.OTHERS_idx = OTHERS_index
        self.w_a = aspect_weight
        self.w_p = polarity_weight

        # self.aspect_fn = MultiLabelFocalLoss(
        #     alpha=aspect_alpha, 
        #     gamma=aspect_gamma,
        # )
        self.aspect_fn = nn.BCEWithLogitsLoss()
        self.polarity_fn = nn.CrossEntropyLoss(reduction="none")
    
    def remove_OTHERS(self, a: torch.Tensor):
        OTHERS_idx = 10 if self.OTHERS_idx is None \
                    else self.OTHERS_idx

        a = torch.cat([
            a[..., 0 : OTHERS_idx],
            a[..., OTHERS_idx+1 : 11]
        ], dim=-1)
        return a

    def forward(
        self,
        y_hat: Tuple[torch.Tensor, torch.Tensor],
        y: Tuple[torch.Tensor, torch.Tensor]
    ):
        a_hat, p_hat = y_hat # Nx(A+1), NxAxP
        a, p = y # Nx(A+1), NxA

        a_loss = self.aspect_fn(a_hat, a)
        a = self.remove_OTHERS(a) # NxA
        
        p_loss = self.polarity_fn(p_hat.transpose(-1, -2), p) * a
        p_loss = p_loss.mean()

        loss = self.w_a * a_loss + self.w_p * p_loss
        return loss


class ViSFDLoss(nn.Module):
    def __init__(
        self, 
        task_type: Literal["stl", "mtl"] = "stl",
        task_loss_args: dict[str, Any] = {},
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.task_type = task_type

        if task_type == "stl":
            self.loss_fn = STL_ViSFDLoss(**task_loss_args)
        elif task_type == "mtl":
            self.loss_fn = MTL_ViSFDLoss(**task_loss_args)
    
    def forward(
        self, 
        y_hat: Tuple[torch.Tensor, torch.Tensor],
        y: Tuple[torch.Tensor, torch.Tensor]
    ):
        return self.loss_fn(y_hat, y)