## Cấu trúc thư mục

#### Tải các file liên quan ở [đây](https://drive.google.com/drive/folders/111ThBNm1B744V5WnsQDU64ou5oAyK9j_?usp=sharing)
#### Để pretrained model, tokenizer, args, trong folder `./baseline`



#### Folder `model` chưa cấu trúc mô hình NER



## Chạy NER:

```
from inference import Inference

ner = Inference('./baseline')

# Input
text = 'ung thư điều trị như nào?'

# Output
output = ner.inference(text)
print(output)
# dict({
#   'ung thư': ['treatment] 
#})
```