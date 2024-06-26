# Aspect-based Sentiment Analysis for Smartphone Feedbacks

Reimplementation, plus trying new architecture according to [this paper](https://link.springer.com/chapter/10.1007/978-3-030-82147-0_53?fbclid=IwAR00G3h4feqS5m_hu8lMbwLw22bXqOjBLrpBzs25eszMN9d7UPjjaCTEcpw). The procedures to use the pretrained model is as the following

```python
from transformers import AutoTokenizer, AutoModel

# Load 2 core modules
tokenizer = AutoTokenizer.from_pretrained("ptdat/vn-smartphone-absa", use_fast=False, trust_remote_code=True)
model = AutoModel.from_pretrained("ptdat/vn-smartphone-absa", trust_remote_code=True)

feedbacks = [
    "Điện thoại tốt, giá phải chăng",
    "Màn hình đẹp, bin trâu nhưng hơi lắc"
]

# Tokenize feedbacks
tokens = tokenizer(feedbacks, padding=True, return_tensors="pt")

# Feed into model
model(**tokens)
```

The results:

```sh
[{'PRICE': 'Positive', 'GENERAL': 'Positive'},
 {'BATTERY': 'Positive', 'PERFORMANCE': 'Negative', 'SCREEN': 'Negative'}]
```
