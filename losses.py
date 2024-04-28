import torch
from torch import nn

from typing import Tuple, Optional

class STL_ViSFDLoss(nn.Module):
    def __init__(
        self, 
        aspect_weight: float = 1.,
        OTHERS_weight: float = 1.,
        include_OTHERS: bool = True,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.include_OTHERS = include_OTHERS
        self.w_a = aspect_weight
        self.w_OTHERS = OTHERS_weight

        self.aspect_fn = nn.CrossEntropyLoss()
        if include_OTHERS:
            self.OTHERS_fn = nn.BCELoss()

    def forward(
        self, 
        y_hat: Tuple[torch.Tensor, torch.Tensor], 
        y: Tuple[torch.Tensor, torch.Tensor]
    ):
        a_hat, o_hat = y_hat
        a, o = y

        a_loss = self.w_a * self.aspect_fn(a_hat, a)
        if not self.include_OTHERS:
            return a_loss
        
        OTHERS_loss = self.w_OTHERS * self.OTHERS_fn(o_hat, o)
        loss = a_loss + OTHERS_loss
        return loss


class MTL_ViSFDLoss(nn.Module):
    def __init__(
        self, 
        aspect_weight: float = 1.,
        polarity_weight: float = 1.,
        aspect_threshold: float = 0.5,
        exclude_OTHERS: bool = True,
        OTHERS_index: Optional[int] = None,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.exclude_OTHERS = exclude_OTHERS
        self.OTHERS_idx = OTHERS_index
        self.w_a = aspect_weight
        self.w_p = polarity_weight
        self.a_threshold = aspect_threshold

        self.aspect_fn = nn.BCELoss()
        self.polarity_fn = nn.CrossEntropyLoss(reduction="none")
    
    def remove_OTHERS(self, a: torch.Tensor):
        a_dim = a.size(-1)
        OTHERS_idx = a_dim if self.OTHERS_idx is None \
                    else self.OTHERS_idx

        if a.ndim == 1:
            a = torch.cat([
                a[0 : OTHERS_idx],
                a[OTHERS_idx+1 : a_dim]
            ])
        elif a.ndim == 2:
            a = torch.cat([
                a[:, 0 : self.OTHERS_idx],
                a[:, OTHERS_idx+1 : a_dim]
            ])
        return a
    
    def normalize_size(
        self,
        y_hat: Tuple[torch.Tensor, torch.Tensor],
        y: Tuple[torch.Tensor, torch.Tensor]
    ):
        a_hat, p_hat = y_hat # (NxA, NxAxP) or (A, AxP)
        a, p = y # (NxA, NxA) or (A, A)

        if p_hat.ndim == 3:
            p_hat = p_hat.transpose(-1, -2) # NxPxA
        elif p_hat.ndim == 2:
            p_hat = p_hat.T.unsqueeze(0) # 1xPxA
            p = p.unsqueeze(0) # 1xA
        
        y_hat = a_hat, p_hat # (NxA, NxPxA) or (A, 1xPxA)
        y = a, p # (NxA, NxA) or (A, A)
        return y_hat, y

    def forward(
        self,
        y_hat: Tuple[torch.Tensor, torch.Tensor],
        y: Tuple[torch.Tensor, torch.Tensor]
    ):
        (a_hat, p_hat), (a, p) = self.normalize_size(y_hat, y)
        # a_hat: Nx(A+1) or A+1
        # p_hat: NxPxA or 1xPxA
        # a: Nx(A+1) or A+1
        # p: NxA or 1xA

        a_loss = self.aspect_fn(a_hat, a)
        if self.exclude_OTHERS:
            a = self.remove_OTHERS(a) # NxA or A
        
        p_loss = self.polarity_fn(p_hat, p) * a
        p_loss = p_loss.mean()

        loss = self.w_a * a_loss + self.w_p * p_loss
        return loss