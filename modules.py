import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

from typing import Optional


class AspectClassifier(nn.Module):
    def __init__(
        self, 
        input_size: int,
        dropout: float = 0.3,
        hidden_size: int = 64,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                in_features=input_size,
                out_features=hidden_size
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                in_features=hidden_size,
                out_features=10+1
            )
        )
    
    def forward(self, input: torch.Tensor):
        x = self.fc(input)
        return x


class PolarityClassifier(nn.Module):
    def __init__(
        self, 
        input_size: int,
        dropout: float = 0.5,
        hidden_size: int = 64,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.polarity_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(
                    in_features=input_size,
                    out_features=hidden_size
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(
                    in_features=hidden_size,
                    out_features=3
                )
            )
            for _ in torch.arange(10)
        ])

    def forward(self, input: torch.Tensor):
        polarities = torch.stack([
            fc(input)
            for fc in self.polarity_fcs
        ])
        
        if input.ndim == 2:
            polarities = polarities.transpose(0, 1)
        return polarities


class LSTM_CNN(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        embed_dim: int = 768,
        dropout: float = 0.3,
        lstm_hidden_size: int = 768,
        lstm_num_layers: int = 2,
        cnn_kernel_size: int = 3,
        cnn_out_channels: int = 16,
        pooling_out_size: int = 8,
        fc_hidden_size: int = 64,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )
        self.conv = nn.Conv1d(
            in_channels=2*lstm_hidden_size,
            out_channels=cnn_out_channels,
            kernel_size=cnn_kernel_size,
        )
        self.avg_pooling = nn.AdaptiveAvgPool1d(pooling_out_size)
        self.max_pooling = nn.AdaptiveMaxPool1d(pooling_out_size)

        self.a_fc = AspectClassifier(
            input_size=cnn_out_channels*pooling_out_size*2,
            hidden_size=fc_hidden_size,
            dropout=dropout
        )
        self.p_fc = PolarityClassifier(
            input_size=cnn_out_channels*pooling_out_size*2,
            hidden_size=fc_hidden_size,
            dropout=dropout
        )

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ):
        x_lens = torch.sum(attention_mask, dim=1).cpu()

        x = self.embed(input_ids)
        x = rnn.pack_padded_sequence(
            x,
            lengths=x_lens,
            batch_first=True,
            enforce_sorted=False
        )
        x, _ = self.lstm(x)
        x, _ = rnn.pad_packed_sequence(
            x,
            batch_first=True,
            total_length=max(self.conv.kernel_size[0], x_lens.max())
        )
        x = self.conv(x.transpose(-1, -2))
        x_avg = self.avg_pooling(x).flatten(start_dim=-2)
        x_max = self.max_pooling(x).flatten(start_dim=-2)
        x = torch.cat([x_avg, x_max], dim=-1)

        a_logits = self.a_fc(x)
        p_logits = self.p_fc(x)
        return a_logits, p_logits


class SmartphoneBERT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_encoders: int = 4,
        encoder_dropout: float = 0.1,
        fc_dropout: float =0.4,
        fc_hidden_size: int = 128,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim,
                dropout=encoder_dropout,
                batch_first=True
            ),
            num_layers=num_encoders,
            norm=nn.LayerNorm(embed_dim),
            enable_nested_tensor=False
        )
        self.a_fc = AspectClassifier(
            input_size=2*embed_dim,
            dropout=fc_dropout,
            hidden_size=fc_hidden_size
        )
        self.p_fc = PolarityClassifier(
            input_size=2*embed_dim,
            dropout=fc_dropout,
            hidden_size=fc_hidden_size
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ):
        padding_mask = ~attention_mask.bool()
        x = self.embed(input_ids)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x[padding_mask] = 0
        x = torch.cat([
            x[..., 0, :],
            torch.mean(x, dim=-2)
        ], dim=-1)

        a_logits = self.a_fc(x)
        p_logits = self.p_fc(x)
        return a_logits, p_logits