import torch
from torch import nn
import torch.nn.utils.rnn as rnn
from tokenizers import Tokenizer

from ..modules import AspectClassifier
from typing import Literal, Sequence, get_args

class PositionalEncoder(nn.Module):
    def __init__(
        self, 
        embed_dim: int, 
        n: int = 1e4,
        max_len: int = 1000
    ):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1).expand(-1, embed_dim)
        i = torch.arange(0, embed_dim, 2).unsqueeze(0).repeat_interleave(2, dim=-1)
        div_term = n ** (2*i / embed_dim)

        pe = position / div_term
        torch.sin_(pe[:, 0::2])
        torch.cos_(pe[:, 1::2])
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        pe = self.pe[: x.size(-2)]
        if x.ndim == 3:
            pe = pe.unsqueeze(0)
        x = x + pe
        return x


class AttentionBlock(nn.Module):
    def __init__(
        self, 
        embed_dim: int = int,
        num_heads: int = 8,
        kdim: int | None = None,
        vdim: int | None = None,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=kdim,
            vdim=vdim,
            batch_first=True
        )
    
    def create_key_padding_mask(self, seq_lens: torch.Tensor):
        batch_size = seq_lens.size(0)
        max_len = seq_lens.max()
        mask = torch.arange(max_len).expand(batch_size, -1).T >= seq_lens
        return mask.T
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_lens: torch.Tensor
    ):
        padding_mask = self.create_key_padding_mask(seq_lens)
        attn, weights = self.attention(
            q, k, v, 
            key_padding_mask=padding_mask
        )
        return q + attn


class PolarityClassifier_v2(nn.Module):
    def __init__(
        self, 
        num_polarities: int = 3,
        embed_dim: int = 768,
        attn_num_heads: int = 8,
        dropout: float = 0.5,
        mlp_hidden_size: int = 128,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.aspect_embed = nn.Embedding(11, embed_dim, padding_idx=10)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.attention = AttentionBlock(
            embed_dim,
            num_heads=attn_num_heads
        )
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, num_polarities)
        )

    def forward(
        self, 
        aspects: torch.Tensor,
        tokens: torch.Tensor,
        seq_lens: torch.Tensor
    ):
        tokens = self.pos_encoder(tokens)

        x = self.aspect_embed(aspects)
        x = self.attention(x, tokens, tokens, seq_lens)
        logits = self.mlp(x)

        return logits


class MTL_ViSFDClassifier_v2(nn.Module):
    def __init__(
        self, 
        input_size: int,
        num_aspects: int = 10,
        num_polarities: int = 3,
        p_embed_dim: int = 768,
        p_attn_num_heads: int = 8,
        dropout: float =  0.4,
        hidden_size: int = 64,
        aspect_threshold: float = 0.5,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self. a_thresh = aspect_threshold

        self.aspect_clf = AspectClassifier(
            input_size=input_size,
            dropout=dropout,
            hidden_size=hidden_size,
            num_aspects=num_aspects
        )
        self.polarity_clf = PolarityClassifier_v2(
            embed_dim=p_embed_dim,
            attn_num_heads=p_attn_num_heads,
            dropout=dropout,
            mlp_hidden_size=hidden_size,
            num_polarities=num_polarities
        )
    

    def aspect_label_to_embed_key(self, aspect_labels: torch.Tensor):
        keys = [
            torch.tensor([i for i, label in enumerate(sample) if label.sigmoid() >= self.a_thresh])
            for sample in aspect_labels
        ]

        return rnn.pad_sequence(keys, batch_first=True, padding_value=10)
    
    def forward(
        self, 
        input: torch.Tensor, 
        tokens: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        aspects = self.aspect_clf(input)
        a_keys = self.aspect_label_to_embed_key(aspects)
        polarities = self.polarity_clf(a_keys, tokens, seq_lens)
        return aspects, polarities
    

class ViSFD_LSTM_CNN_v2(nn.Module):
    def __init__(
        self, 
        # vocab_size: int,
        tokenizer: Tokenizer,
        num_aspects: int = 10,
        num_polarities: int = 3,
        embed_dim: int = 768,
        dropout: float = 0.2,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        cnn_kernel_size: int = 3,
        cnn_out_channels: int = 16,
        pooling_out_size: int = 8,
        output_dropout: float = 0.3,
        output_hidden_size: int = 64,
        output_attn_num_heads: int = 8,
        aspect_threshold: float = 0.5,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

        self.embedding = nn.Embedding(
            num_embeddings=tokenizer.get_vocab_size(),
            embedding_dim=embed_dim,
            padding_idx=0,
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )
        self.cnn = nn.Conv1d(
            in_channels=2*lstm_hidden_size,
            out_channels=cnn_out_channels,
            kernel_size=cnn_kernel_size,
        )

        self.avg_pooling = nn.AdaptiveAvgPool1d(pooling_out_size)
        self.max_pooling = nn.AdaptiveMaxPool1d(pooling_out_size)

        self.output_layer = MTL_ViSFDClassifier_v2(
            input_size=cnn_out_channels*pooling_out_size*2,
            num_aspects=num_aspects,
            num_polarities=num_polarities,
            p_embed_dim=embed_dim,
            p_attn_num_heads=output_attn_num_heads,
            dropout=output_dropout,
            hidden_size=output_hidden_size,
            aspect_threshold=aspect_threshold
        )
    
    def forward(self, input: str | Sequence[str], return_a_padding_mask: bool = True):
        if isinstance(input, str):
            input = [input]

        encodings = self.tokenizer.encode_batch(input)
        x = [
            torch.tensor(encoding.ids)
            for encoding in encodings
        ]
        x = rnn.pad_sequence(x, batch_first=True)
        x_lens = torch.tensor([
            *map(len, encodings)
        ], device="cpu")

        tokens = self.embedding(x)

        x = rnn.pack_padded_sequence(tokens, x_lens, batch_first=True, enforce_sorted=False)
        x, (h, c) = self.lstm(x)
        x, seq_lens = rnn.pad_packed_sequence(x, batch_first=True)
        
        x = self.cnn(x.transpose(-1, -2))
        x_avg = self.avg_pooling(x).flatten(start_dim=-2)
        x_max = self.max_pooling(x).flatten(start_dim=-2)
        x = torch.cat([x_avg, x_max], dim=-1)

        if self.task_type == "stl":
            x = self.output_layer(x)
        elif self.task_type == "mtl":
            x = self.output_layer(x, tokens, seq_lens, return_a_padding_mask=return_a_padding_mask)

        return x