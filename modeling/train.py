import torch
from modules import ViSFD_LSTM
from losses import MTL_ViSFDLoss, STL_ViSFDLoss
from dataset import ViSFDDataset
from tokenizers import Tokenizer

import json
from utils import train

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    train_set = ViSFDDataset("data/Train.csv")
    val_set = ViSFDDataset("data/Dev.csv")

    tokenizer: Tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
    model = ViSFD_LSTM(
        tokenizer
    )

    train(
        model,
        STL_ViSFDLoss(),
        torch.optim.AdamW(model.parameters()),
        train_set,
        val_set
    )