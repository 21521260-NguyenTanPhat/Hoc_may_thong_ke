import torch
from torch.utils.data import Dataset

from pandas import read_csv
from pathlib import Path
from typing import Literal
import re

class ViSFD(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "dev", "test"]
    ) -> None:
        super().__init__()
        root_dir = Path(root_dir)
        split = split.capitalize()

        self.root_dir = root_dir
        self.split = split

        self.ASPECT_LOOKUP = {
            a: i
            for i, a in enumerate(["CAMERA", "FEATURES", "BATTERY", "PRICE", "GENERAL", "SER&ACC", "PERFORMANCE", "SCREEN", "DESIGN", "STORAGE", "OTHERS"])
        }
        self.POLARITY_LOOKUP = {
            p: i
            for i, p in enumerate(["Negative", "Neutral", "Positive"])
        }

        self.data = (
            read_csv(root_dir / f"{split}.csv") \
            .set_index("index")
        )
        with open(root_dir / f"prep-{split}.txt", "r") as f:
            self.data["preprocess"] = f.read().splitlines()

        for a in self.ASPECT_LOOKUP.keys():
            self.data[a] = self.data["label"].map(lambda label: self.get_sentiment(label, a))
        self.label_frequencies = torch.tensor((self.data[list(self.ASPECT_LOOKUP.keys())] > -1).values).sum(dim=0)
    
    def get_sentiment(self, label: str, aspect: str):
        if aspect == "OTHERS":
            return 1 if "{OTHERS}" in label else -1
        
        p = re.search(rf"{{{aspect}#(\w+)}}", label)
        if p is None:
            return -1
        p = p.group(1)
        return self.POLARITY_LOOKUP[p]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        text = self.data.loc[index, "preprocess"]
        aspects = (self.data.loc[index, list(self.ASPECT_LOOKUP.keys())] >= 0).values.astype("float")
        polarities = self.data.loc[index, list(self.ASPECT_LOOKUP.keys())[:-1]].values.astype("long")

        return {
            "text": text,
            "aspects": torch.from_numpy(aspects),
            "polarities": torch.from_numpy(polarities)
        }