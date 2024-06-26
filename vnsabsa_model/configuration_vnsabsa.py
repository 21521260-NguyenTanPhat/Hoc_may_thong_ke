from transformers import PretrainedConfig

class VnSmartphoneAbsaConfig(PretrainedConfig):
    model_type = "vnsabsa"
    def __init__(
        self, 
        vocab_size: int = 5272,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_encoders: int = 4,
        encoder_dropout: float = 0.1,
        fc_dropout: float =0.4,
        fc_hidden_size: int = 128,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_encoders = num_encoders
        self.encoder_dropout = encoder_dropout
        self.fc_dropout = fc_dropout
        self.fc_hidden_size = fc_hidden_size

        super().__init__(**kwargs)