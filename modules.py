import torch
from torch import nn

from typing import get_args, Literal

# =============== OUTPUT LAYERS ===============

class AspectClassifier(nn.Module):
    def __init__(
        self, 
        input_size: int,
        num_aspects: int = 10,
        include_others: bool = True,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.input_size = input_size
        self.num_aspects = num_aspects
        self.include_others = include_others

        self.fc = nn.Linear(
            in_features=input_size,
            out_features=num_aspects+include_others
        )
    
    def forward(self, input: torch.Tensor):
        x = self.fc(input)
        return x


class PolarityClassifier(nn.Module):
    def __init__(
        self, 
        input_size: int,
        num_aspects: int = 10,
        num_polarities: int = 3,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.polarity_fcs = nn.ModuleList([
            nn.Linear(
                in_features=input_size,
                out_features=num_polarities
            ) for _ in torch.arange(num_aspects)
        ])

    def forward(self, input: torch.Tensor):
        polarities = torch.stack([
            fc(input)
            for fc in self.polarity_fcs
        ])
        
        if input.ndim == 2:
            polarities = polarities.transpose(0, 1)
        return polarities

    def predict(self, input: torch.Tensor):
        x = self(input)
        return x.argmax(dim=-1).to(dtype=torch.int8)
    

class MTL_ViSFDClassifier(nn.Module):
    def __init__(
        self, 
        input_size: int,
        num_aspects: int = 10,
        num_polarities: int = 3,
        include_OTHERS: bool = True,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.aspect_clf = AspectClassifier(
            input_size=input_size,
            num_aspects=num_aspects,
            include_others=include_OTHERS,
        )
        self.polarity_clf = PolarityClassifier(
            input_size=input_size,
            num_aspects=num_aspects,
            num_polarities=num_polarities
        )
    
    def forward(self, input: torch.Tensor):
        aspects = self.aspect_clf(input)
        polarities = self.polarity_clf(input)
        return aspects, polarities


class STL_ViSFDClassifier(nn.Module):
    def __init__(
        self, 
        input_size: int,
        num_aspects: int = 10,
        num_polarities: int = 3,
        include_OTHERS: bool = True,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.include_OTHERS = include_OTHERS

        self.clf_list = nn.ModuleList([
            nn.Linear(input_size, num_polarities+1)
            for i in torch.arange(num_aspects)
        ])
        self.OTHERS_clf = nn.Linear(input_size, 1) if include_OTHERS else None
    
    def forward(self, input: torch.Tensor):
        result = torch.stack([
            clf(input)
            for clf in self.clf_list
        ], dim=-2)

        if not self.include_OTHERS:
            return result
        
        result_OTHERS = self.OTHERS_clf(input).sigmoid()
        return result, result_OTHERS


class ViSFD_LSTM(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        num_aspects: int = 10,
        num_polarities: int = 3,
        include_OTHERS: bool = True,
        task_type: Literal["stl", "mtl"] = "stl",
        embed_dim: int = 768,
        dropout: float = 0.2,
        lstm_hidden_size: int = 128,
        cnn_kernel_size: int = 3,
        cnn_out_channels: int = 16,
        pooling_out_size: int = 16,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )
        self.dropout = nn.Dropout1d(dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.cnn = nn.Conv1d(
            in_channels=2*lstm_hidden_size,
            out_channels=cnn_out_channels,
            kernel_size=cnn_kernel_size,
        )

        self.avg_pooling = nn.AdaptiveAvgPool1d(pooling_out_size)
        self.max_pooling = nn.AdaptiveMaxPool1d(pooling_out_size)

        if task_type not in get_args(Literal["stl", "mtl"]):
            raise ValueError("Task type not supported, should be either 'stl' or 'mtl'")
        if task_type == "stl":
            self.output_layer = STL_ViSFDClassifier(
                input_size=cnn_out_channels*pooling_out_size*2,
                num_aspects=num_aspects,
                num_polarities=num_polarities,
                include_OTHERS=include_OTHERS
            )
        elif task_type == "mtl":
            self.output_layer = MTL_ViSFDClassifier(
                input_size=cnn_out_channels*pooling_out_size*2,
                num_aspects=num_aspects,
                num_polarities=num_polarities,
                include_OTHERS=include_OTHERS,
            )
    
    def forward(self, input: torch.Tensor):
        x = self.embedding(input)
        x = self.dropout(x.transpose(-1, -2)).transpose(-1, -2)
        x, (h, c) = self.lstm(x)
        x = self.cnn(x.transpose(-1, -2))
        x_avg = self.avg_pooling(x).flatten(start_dim=-2)
        x_max = self.max_pooling(x).flatten(start_dim=-2)
        x = torch.cat([x_avg, x_max], dim=-1)

        x = self.output_layer(x)
        return x
    

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
        if x.ndim == 3:
            pe = self.pe[:x.size(1)].unsqueeze(0)
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
        

class ViSFD_Attention(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        num_aspects: int = 10,
        num_polarities: int = 3,
        include_OTHERS: bool = True,
        task_type: Literal["stl", "mtl"] = "stl",
        embed_size: int = 768,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            padding_idx=0
        )
        self.pos_encoder = PositionalEncoder(
            embed_dim=embed_size,
            max_len=500
        )