import pandas as pd
from py_vncorenlp import VnCoreNLP

from langdetect import detect
import os
import argparse
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("-o", "--out_dir", default="./tokenizer")
    args = vars(parser.parse_args())

    csv_path = args["csv_path"]
    out_dir = args["out_dir"]

    visfd = pd.read_csv(csv_path).set_index("index")
    visfd["lang"] = visfd.comment.map(detect)

    working_dir = os.getcwd()
    vncorenlp = VnCoreNLP(save_dir=os.environ["VNCORENLP"])
    os.chdir(working_dir)

    comments = tqdm(
        visfd.comment[visfd.lang == "vi"].values,
        desc="Progress",
        total=visfd.shape[0],
        colour="green"
    )
    feedbacks = [" ".join(vncorenlp.word_segment(c)) for c in comments]

    with open(os.path.join(out_dir, "feedbacks.txt"), 'w') as f:
        for feedback in feedbacks:
            f.write(f"{feedback}\n")