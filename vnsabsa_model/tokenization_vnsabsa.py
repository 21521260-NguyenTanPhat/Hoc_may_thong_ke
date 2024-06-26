import torch
from typing import Dict
from transformers import PreTrainedTokenizer
from tokenizers.implementations import CharBPETokenizer
from tokenizers.processors import TemplateProcessing
import regex as re

from typing import Tuple, Optional
import shutil
import os
import requests

class VnSmartphoneAbsaTokenizer(PreTrainedTokenizer):
    vocab_files_names = {
        "vocab_file": "vocab.txt",
        "merge_file": "merge.txt",
    }
    pretrained_vocab_files_map = {
        "vocab_file": "https://huggingface.co/ptdat/vn-smartphone-absa/resolve/main/vocab.txt",
        "merge_file": "https://huggingface.co/ptdat/vn-smartphone-absa/resolve/main/merge.txt"
    }
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self, 
        vocab_file, 
        merge_file, 
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs
    ):
        self.vocab_file = vocab_file
        self.merge_file = merge_file

        self.tokenizer = CharBPETokenizer(vocab_file, merge_file, lowercase=True, bert_normalizer=False, split_on_whitespace_only=True)
        self.tokenizer.post_processor = TemplateProcessing(
            single="<s> $9 </s>",
            pair="<s> $A </s> $B:1 </s>:1",
            special_tokens=[
                ("<s>", 2),
                ("</s>", 3)
            ]
        )
        self.tokenizer.enable_padding(pad_token="<pad>")
        
        self.encoder = self.tokenizer.get_vocab()
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        self.prepare_preprocess()

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs
        )
    
    def _tokenize(self, text: str):
        text = self.normalize(text)
        return self.tokenizer.encode(text).tokens
    
    def get_vocab(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()
    
    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.txt"
        )
        out_merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + "merge.txt"
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            shutil.copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        if os.path.abspath(self.merge_file) != os.path.abspath(out_merge_file):
            shutil.copyfile(self.merge_file, out_merge_file)

        return out_vocab_file, out_merge_file
    
    def _convert_token_to_id(self, token: str):
        return self.encoder.get(token, self.encoder[self.unk_token])
    
    def _convert_id_to_token(self, id: int):
        return self.decoder.get(id, self.unk_token) 
    
    def prepare_preprocess(self):
        self.uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
        self.unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

        self.dict_char = {}
        char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
            '|')
        charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
            '|')
        for i in range(len(char1252)):
            self.dict_char[char1252[i]] = charutf8[i]

        self.bang_nguyen_am = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
                        ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
                        ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
                        ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
                        ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
                        ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
                        ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
                        ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
                        ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
                        ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
                        ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
                        ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]
        self.bang_ky_tu_dau = ['', 'f', 's', 'r', 'x', 'j']

        self.nguyen_am_to_ids = {}

        for i in range(len(self.bang_nguyen_am)):
            for j in range(len(self.bang_nguyen_am[i]) - 1):
                self.nguyen_am_to_ids[self.bang_nguyen_am[i][j]] = (i, j)

        self.sp_word_sub = {
            "@@": "confuseeyes",
            "℅": "%",
            r"/": " fraction ",
            r":\)+": "smileface",
            r";\)+": "smileface",
            r":\*+": "kissingface",
            r"=\)+": "playfulsmileface",
            r"=\(+": "playfulsadface",
            r":\(+": "sadface",
            r":3+": "threeface",
            r":v+": "vface",
            r"\^\^": "kindsmile",
            r"\^_\^": "kindmountsmile",
            r"\^\.\^": "kindmountsmile",
            r"-_-": "disapointface",
            r"\._\.": "confusedface",
            r":>+": "cutesmile",
            r"(\|)w(\|)": "fancycryface",
            r":\|": "mutedface",
            r":d+": "laughface",
            r"<3": "loveicon",
            r"\.{2,}": "threedot",
            r"-{1,}>{1,}": "arrow",
            r"={1,}>{1,}": "arrow",
            r"(\d+)h": r"\1 giờ",
            r"(\d+)'": r"\1 phút",
            r"(\d+)trieu": r"\1 triệu",
            r"(\d+)\s?tr": r"\1 triệu",
            r"blut\w+": "bluetooth",
            r"(\d+)\s\*": r"\1 sao"
        }

        self.replace_dict = {
            "/": "fraction",
            "wf": "wifi",
            "wifj": "wifi",
            "wjfj": "wifi",
            "wjfi": "wifi",
            "wiffi": "wifi",
            "wj": "wifi",
            "ko": "không",
            "k": "không",
            "hong": "không",
            "đc": "được",
            "sp": "sản phẩm",
            "fb": "facebook",
            "ytb": "youtube",
            "yt": "youtube",
            "mes": "messenger",
            "mess": "messenger",
            "tgdđ": "thegioididong",
            "nv": "nhân viên",
            "ss": "samsung",
            "ip": "iphone",
            "appel": "apple",
            "oke": "ok",
            "okie": "ok",
            "okey": "ok",
            "oki": "ok",
            "oce": "ok",
            "okela": "ok",
            "mk": "mình",
            "sd": "sử dụng",
            "sdung": "sử dụng",
            "ae": "anh em",
            "lq": "liên quân",
            "lqmb": "liên quân mobile",
            "lun": "luôn",
            "ng": "người",
            "ad": "admin",
            "ms": "mới",
            "cx": "cũng",
            "cũg": "cũng",
            "nhìu": "nhiều",
            "bth": "bình thường",
            "bthg": "bình thường",
            "ngta": "người ta",
            "dow": "download",
            "hdh": "hệ điều hành",
            "hđh": "hệ điều hành",
            "cammera": "camera",
            "dt": "điện thoại",
            "dthoai": "điện thoại",
            "dth": "điện thoại",
            "đth": "điện thoại",
            "hk": "không",
            "j": "gì",
            "ji": "gì",
            "mn": "mọi người",
            "m.n": "mọi người",
            "mjh": "mình",
            "mjk": "mình",
            "lắc": "lag",
            "lác": "lag",
            "lang": "lag",
            "nhah": "nhanh",
            "nóichung": "nói chung",
            "zl": "zalo",
            "sóg": "sóng",
            "rẽ": "rẻ",
            "trc": "trước",
            "chíp": "chip",
            "bin": "pin",
            "lm": "làm",
            "bik": "biết",
            "hog": "không",
            "zỏm": "dổm",
            "z": "vậy",
            "v": "vậy",
            "nhah": "nhanh",
            "r": "rồi",
            "ỗn": "ổn",
            "nhìu": "nhiều",
            "wá": "quá",
            "wep": "web",
            "wed": "web",
            "fim": "phim",
            "film": "phim",
            "xạc": "sạc",
            "xài": "sài",
            "het": "hết",
            "lun": "luôn",
            "e": "em",
            "a": "anh",
            "bjo": "bây giờ",
            "vl": "vãi lồn",
            "sac": "sạc",
            "vidieo": "video",
            "tét": "test",
            "tes": "test",
            "thik": "thích",
            "fai": "phải",
            "✋": "tay",
            "🔋": "pin",
            "☆": "sao",
            "supper": "super",
            "lổi": "lỗi",
            "loát": "load",
            "thui": "thôi",
            "rùi": "rồi",
            "ỗn": "ổn",
            "lổi": "lỗi",
            "suống": "xuống",
            "selfi": "selfie",
            "gg": "google",
            "cam on": "cảm ơn",
            "tg": "thời gian",
            "nchung": "nói chung",
            "❤": "loveicon",
            "trại nghiệm": "trải nghiệm",
            "dất": "rất",
            "đứg": "đứng",
            "bằg": "bằng",
            "mìh": "mình",
            "đag": "đang",
            "thoi": "thôi",
            "củng": "cũng",
            "đả": "đã",
            "màng": "màn",
            "ff": "free fire",
            "cod": "call of duty",
            "moi thứ": "mọi thứ",
            "moi thu": "mọi thứ",
            "moi thư": "mọi thứ",
            "moi người": "mọi người",
            "moi": "mới",
            "dk": "được",
            "đk": "được",
            "nhậy": "nhạy",
            "ak": "á",
            "ghe": "nghe",
            "bùn": "buồn",
            "bit": "biết",
            "bít": "biết",
            "bnhieu": "bao nhiêu",
            "dụg": "dụng",
            "tk": "tài khoản",
            "sąc": "sạc",
            "rât": "rât",
            "haz": "haiz",
            "sai làm": "sai lầm",
            "flim": "film",
            "xướt": "xước",
            "viềng": "viền"
        }

    def convert_unicode(self, text: str):
        return re.sub(
            r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
            lambda x: self.dict_char[x.group()], text
        )
    
    def is_valid_vietnam_word(self, word):
        chars = list(word)
        nguyen_am_index = -1
        for index, char in enumerate(chars):
            x, y = self.nguyen_am_to_ids.get(char, (-1, -1))
            if x != -1:
                if nguyen_am_index == -1:
                    nguyen_am_index = index
                else:
                    if index - nguyen_am_index != 1:
                        return False
                    nguyen_am_index = index
        return True
    
    def chuan_hoa_dau_tu_tieng_viet(self, word):
        if not self.is_valid_vietnam_word(word):
            return word

        chars = list(word)
        dau_cau = 0
        nguyen_am_index = []
        qu_or_gi = False
        for index, char in enumerate(chars):
            x, y = self.nguyen_am_to_ids.get(char, (-1, -1))
            if x == -1:
                continue
            elif x == 9:  # check qu
                if index != 0 and chars[index - 1] == 'q':
                    chars[index] = 'u'
                    qu_or_gi = True
            elif x == 5:  # check gi
                if index != 0 and chars[index - 1] == 'g':
                    chars[index] = 'i'
                    qu_or_gi = True
            if y != 0:
                dau_cau = y
                chars[index] = self.bang_nguyen_am[x][0]
            if not qu_or_gi or index != 1:
                nguyen_am_index.append(index)
        if len(nguyen_am_index) < 2:
            if qu_or_gi:
                if len(chars) == 2:
                    x, y = self.nguyen_am_to_ids.get(chars[1])
                    chars[1] = self.bang_nguyen_am[x][dau_cau]
                else:
                    x, y = self.nguyen_am_to_ids.get(chars[2], (-1, -1))
                    if x != -1:
                        chars[2] = self.bang_nguyen_am[x][dau_cau]
                    else:
                        chars[1] = self.bang_nguyen_am[5][dau_cau] if chars[1] == 'i' else self.bang_nguyen_am[9][dau_cau]
                return ''.join(chars)
            return word

        for index in nguyen_am_index:
            x, y = self.nguyen_am_to_ids[chars[index]]
            if x == 4 or x == 8:  # ê, ơ
                chars[index] = self.bang_nguyen_am[x][dau_cau]
                # for index2 in nguyen_am_index:
                #     if index2 != index:
                #         x, y = nguyen_am_to_ids[chars[index]]
                #         chars[index2] = bang_nguyen_am[x][0]
                return ''.join(chars)

        if len(nguyen_am_index) == 2:
            if nguyen_am_index[-1] == len(chars) - 1:
                x, y = self.nguyen_am_to_ids[chars[nguyen_am_index[0]]]
                chars[nguyen_am_index[0]] = self.bang_nguyen_am[x][dau_cau]
                # x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
                # chars[nguyen_am_index[1]] = bang_nguyen_am[x][0]
            else:
                # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
                # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
                x, y = self.nguyen_am_to_ids[chars[nguyen_am_index[1]]]
                chars[nguyen_am_index[1]] = self.bang_nguyen_am[x][dau_cau]
        else:
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
            x, y = self.nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = self.bang_nguyen_am[x][dau_cau]
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[2]]]
            # chars[nguyen_am_index[2]] = bang_nguyen_am[x][0]
        return ''.join(chars)
        
    def chuan_hoa_dau_cau_tieng_viet(self, sentence):
        """
            Chuyển câu tiếng việt về chuẩn gõ dấu kiểu cũ.
            :param sentence:
            :return:
        """
        words = sentence.split()
        for index, word in enumerate(words):
            cw = re.sub(r'(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)', r'\1/\2/\3', word).split('/')
            # print(cw)
            if len(cw) == 3:
                cw[1] = self.chuan_hoa_dau_tu_tieng_viet(cw[1])
            words[index] = ''.join(cw)
        return ' '.join(words)
    
    def normalize(self, text: str, track_change=False):
        # Lowercase
        text = text.lower()

        text = re.sub(r"((https?|ftp|file):\/{2,3})+([-\w+&@#/%=~|$?!:,.]*)|(www.)+([-\w+&@#/%=~|$?!:,.]*)", "urllink", text)

        # Remove dup trailing chars (troiiiii -> troi)
        text = re.sub(r"([\D\w])\1+\b", r"\1", text)
        if track_change:
            print("Dedup trailing: ", text)

        # Replace special symbol to word
        for pttn, repl in self.sp_word_sub.items():
            text = re.sub(fr"{pttn}", f" {repl} ", text)
        if track_change:
            print("Replace special word: ", text)
        
        # Correct misspelled word
        def replace(match):
            orig = match.group(1)
            word = " " + self.replace_dict.get(orig, orig) + " "
            return word
        text = re.sub(r"\b(\S+)\b", replace, text)
        if track_change:
            print("Correct misspelled word: ", text)

        # Normalize string encoding
        text = self.convert_unicode(text)
        if track_change:
            print("Normalize string encoding: ", text)

        # Vietnamese unicode normalization
        text = self.chuan_hoa_dau_cau_tieng_viet(text)
        if track_change:
            print("Vietnamese unicode normalization: ", text)

        # Eliminate decimal delimiter (9.000 -> 9000)
        text = re.sub(r"(?<=\d)\.(?=\d{3})", "", text)
        if track_change:
            print("Eliminate decimal delimiter: ", text)
        
        # Split between value and unit (300km -> 300 km)
        text = re.sub(r"(\d+)(\D+)", r"\1 \2", text)
        if track_change:
            print("Split between value and unit: ", text)

        # Split by punctuations
        text = " ".join(
            re.split("(["+re.escape("!\"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~")+"])", text)
        )
        if track_change:
            print("Split by punctuations: ", text)

        # Split by emoticons
        text = " ".join(
            re.split("(["
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
            "])", text)
        )

        # Word segmentation
        # text = " ".join(vncorenlp.word_segment(text))
        
        return text