import torch
from torch.utils.data import Dataset
from py_vncorenlp import VnCoreNLP

from pandas import read_csv
from langdetect import detect
import re
import os
from tqdm import tqdm
from typing import Optional, Literal

tqdm.pandas(colour="green")
cwd = os.getcwd()
vncorenlp = VnCoreNLP(save_dir=os.environ["VNCORENLP"])
os.chdir(cwd)

class ViSFDDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        OTHERS_idx: Optional[int] = None, # Default to last index (10)
        task_type: Literal["stl", "mtl"] = "stl",
    ) -> None:
        super().__init__()
        self.A_P_REGEX = re.compile(r"{([A-Z&]+)#?(\w+)?}")
        self.ASPECT_LOOKUP = {
            a: i
            for i, a in enumerate(["CAMERA", "FEATURES", "BATTERY", "PRICE", "GENERAL", "SER&ACC", "PERFORMANCE", "SCREEN", "DESIGN", "STORAGE"])
        }
        self.POLARITY_LOOKUP = {
            p: i
            for i, p in enumerate(["Negative", "Neutral", "Positive"])
        }
        self.OTHERS_idx = 10 if OTHERS_idx is None else OTHERS_idx
        self.task_type = task_type

        self.data = (
            read_csv(data_path) \
            .set_index("index")
        )

        self.data["target"] = self.data.label.progress_map(self.label_to_tensor)
        self.data["segmented"] = self.data.comment.progress_map(self.segment)
        self.data["lang"] = self.data.comment.progress_map(detect)

        self.data = self.data[self.data.lang == "vi"]
    
    def segment(self, text: str):
        return " ".join(vncorenlp.word_segment(text))

    def label_to_tensor(self, label: str):
        d = dict(self.A_P_REGEX.findall(label))
        if self.task_type == "stl":
            p_label = torch.zeros(10, dtype=torch.long)
            OTHERS_label = torch.tensor(["OTHERS" in d], dtype=torch.float)
        elif self.task_type == "mtl":
            a_label = torch.zeros(11)
            p_label = torch.zeros(10, dtype=torch.long)

            a_label[self.OTHERS_idx] = 1

        for a, p in d.items():
            if a == "OTHERS":
                continue
            i_a = self.ASPECT_LOOKUP[a]
            i_p = self.POLARITY_LOOKUP[p]

            p_label[i_a] = i_p
            if self.task_type == "mtl":
                a_label[i_a] = 1
        
        if self.task_type == "stl":
            p_label[p_label!=0] += 1
            result = p_label, OTHERS_label
        elif self.task_type == "mtl":
            result = a_label, p_label

        return result
    
    def __len__(self):
        return self.data.target.shape[0]
    
    def __getitem__(self, index: int):
        feedback = self.data.segmented.values[index]
        target = self.data.target.values[index]

        return feedback, target