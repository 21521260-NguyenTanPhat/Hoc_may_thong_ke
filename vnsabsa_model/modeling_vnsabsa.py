from transformers import PreTrainedModel
from modules import SmartphoneBERT
import torch

from .configuration_vnsabsa import VnSmartphoneAbsaConfig

from typing import Tuple


class VnSmartphoneAbsaModel(PreTrainedModel):
    config_class = VnSmartphoneAbsaConfig
    
    def __init__(
        self,
        config: VnSmartphoneAbsaConfig
    ):
        super().__init__(config)
        self.model = SmartphoneBERT(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_encoders=config.num_encoders,
            encoder_dropout=config.encoder_dropout,
            fc_dropout=config.fc_dropout,
            fc_hidden_size=config.fc_hidden_size
        )
        self.ASPECT_LOOKUP = {
            i: a
            for i, a in enumerate(["CAMERA", "FEATURES", "BATTERY", "PRICE", "GENERAL", "SER&ACC", "PERFORMANCE", "SCREEN", "DESIGN", "STORAGE", "OTHERS"])
        }
        self.POLARITY_LOOKUP = {
            i: p
            for i, p in enumerate(["Negative", "Neutral", "Positive"])
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        aspect_thresholds: float | torch.Tensor = 0.5
    ):
        pred = self.model(input_ids, attention_mask)
        result = self.decode_absa(
            pred, 
            aspect_thresholds=aspect_thresholds
        )
        return result

    def decode_absa(
        self, 
        pred: Tuple[torch.Tensor, torch.Tensor],
        aspect_thresholds: float | torch.Tensor = 0.5
    ):
        if isinstance(aspect_thresholds, float):
            aspect_thresholds = torch.full((11,), aspect_thresholds)
        
        a, p = pred
        a = a.sigmoid().cpu()
        p = p.argmax(dim=-1).cpu()

        results = []
        for a_i, p_i in zip(a, p):
            res_i = {}
            for i in range(10):
                a = self.ASPECT_LOOKUP[i]
                p = self.POLARITY_LOOKUP[p_i[i].item()]
                if a_i[i] >= aspect_thresholds[i]:
                    res_i[a] = p
            results.append(res_i)

            # OTHERS
            if a_i[-1] >= aspect_thresholds[-1]:
                res_i["OTHERS"] = ""
        
        return results