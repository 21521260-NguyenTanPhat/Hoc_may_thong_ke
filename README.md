# Aspect-based Sentiment Analysis for Smartphone Feedbacks

The procedures to use the pretrained model is as follow

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
