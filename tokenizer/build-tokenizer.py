import tokenizers
from tokenizers import models
from tokenizers import normalizers
from tokenizers import trainers
from tokenizers import pre_tokenizers
from tokenizers import processors
from tokenizers import decoders

import re
import json
import argparse
import os
from typing import Literal, Any, TypedDict


class TokenizerConfig(TypedDict):
    algo: Literal["bpe", "wordpiece", "word", "unigram"]
    unknown_token: str
    pad_token: str
    bos_token: str
    eos_token: str
    padding: Literal["left", "right"] | None
    byte_level: bool
    bpe_trainer_kwargs: dict[str, Any]
    wordpiece_trainer_kwargs: dict[str, Any]
    word_trainer_kwargs: dict[str, Any]


def extract_modules(config: TokenizerConfig) -> tuple[models.Model, pre_tokenizers.PreTokenizer, decoders.Decoder, trainers.Trainer, dict[str, int]]:
    unk_token = config["unknown_token"]
    algo = config["algo"]
    byte_lvl = config["byte_level"]

    special_tokens = [config[name] for name in ["pad_token", "unknown_token", "bos_token", "eos_token"]]

    if algo == "bpe":
        model = models.BPE(unk_token=unk_token)
        trainer = trainers.BpeTrainer(**config["bpe_trainer_kwargs"], special_tokens=special_tokens)
        decoder = decoders.Sequence([decoders.ByteLevel(), decoders.BPEDecoder()]) if byte_lvl else decoders.BPEDecoder()
    elif algo == "wordpiece":
        model = models.WordPiece(unk_token=unk_token)
        trainer = trainers.WordPieceTrainer(**config["wordpiece_trainer_kwargs"], special_tokens=special_tokens)
        decoder = decoders.Sequence([decoders.ByteLevel(), decoders.WordPiece()]) if byte_lvl else decoders.WordPiece()
    elif algo == "word":
        model = models.WordLevel(unk_token=unk_token)
        trainer = trainers.WordLevelTrainer(**config["word_trainer_kwargs"], special_tokens=special_tokens)
        decoder = decoders.ByteLevel() if byte_lvl else None
    
    pre_tokenizer = pre_tokenizers.ByteLevel() if byte_lvl else pre_tokenizers.WhitespaceSplit()
    return model, pre_tokenizer, decoder, trainer, special_tokens


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data_files", default=[], type=lambda s: [s])
    parser.add_argument("-o", "--out_dir", default="./tokenizer")
    parser.add_argument("-conf", "--config_file", default="../config.json")
    args = vars(parser.parse_args())

    data_files = args["data_files"]
    out_dir = args["out_dir"]
    config_path = args["config_file"]

    
    with open(config_path, "r") as conf_file:
        config = TokenizerConfig(**json.load(conf_file)["tokenizer"])

    model, pre_tokenizer, decoder, trainer, special_tokens = extract_modules(config)


    tokenizer = tokenizers.Tokenizer(model)
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.Strip(),
        normalizers.Replace(r"\s{2,}", " "),
        normalizers.Lowercase()
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(tokenizers.Regex("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u'\U00010000-\U0010ffff'
            u"\u200d"
            u"\u2640-\u2642"
            u"\u2600-\u2B55"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\u3030"
            u"\ufe0f"
            u"\u221a"
        "]"), "isolated"),
        pre_tokenizers.Digits(),
        pre_tokenizers.Split(
            tokenizers.Regex("["+re.escape("!\"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~")+"]"), 
            "isolated"
        ),
        pre_tokenizer
    ])
    tokenizer.post_processor = processors.Sequence([
        processors.TemplateProcessing(
            single=f"{config['bos_token']} $9 {config['eos_token']}",
            pair=f"{config['bos_token']} $A {config['eos_token']} $B:1 {config['eos_token']}:1",
            special_tokens=[
                (config['bos_token'], 2),
                (config['eos_token'], 3)
            ]
        )
    ])
    if decoder:
        tokenizer.decoder = decoder

    if config["padding"] is not None:
        tokenizer.enable_padding(
            direction=config.padding,
            pad_id=0,
            pad_token=config.pad_token
        )


    tokenizer.train(data_files, trainer)
    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))