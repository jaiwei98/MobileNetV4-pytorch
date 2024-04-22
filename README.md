# MobileNetV4-pytorch
An unofficial implementation of MobileNetV4 (MNv4) in Pytorch. <br />

There are 5 types of MNv4 as indicated in the <a href="https://arxiv.org/abs/2404.10518">MobileNetV4 -- Universal Models for the Mobile Ecosystem</a>, e.g. 
- MobileNetV4ConvSmall (MNv4-Conv-S)
- MobileNetV4ConvMedium (MNv4-Conv-M)
- MobileNetV4ConvLarge (MNv4-Conv-L)
- MobileNetV4HybridMedium (MNv4-Hybrid-M)
- MobileNetV4HybridLarge (MNv4-Hybrid-L)

## Table of Content
- [How to initiate model](#mobilenetv4)
- [Some TODO](#todo)
- [Notes](#notes)

## MobileNetV4
This section mainly showed how to import MobileNetV4

```python
import torch
from mobilenet.mobilenetv4 import MobileNetV4

# Support ['MobileNetV4ConvSmall', 'MobileNetV4ConvMedium', 'MobileNetV4ConvLarge']
# Will be supported soon ['MobileNetV4HybridMedium', 'MobileNetV4HybridLarge']
model = MobileNetV4("MobileNetV4ConvSmall")

# Check the trainable params
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

# Check the model's output shape
print("Check output shape ...")
x = torch.rand(1, 3, 224, 224)
y = model(x)
for i in y:
    print(i.shape)
```

## TODO
- [ ] Support 'MobileNetV4HybridMedium' and 'MobileNetV4HybridLarge'
- [ ] Release pretrained weight (welcome any contributors to submit PR)

## Notes
Note that there are few parts which not excatly the same as implementation in tensorflow 
- The "fused_ib" block 
- The global average pooling layers at the end of model

## Credits
Some function and code are adapted and referenced from official repo <a href="https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py">tensorflow/models</a>.

## Citations
```bibtex
@misc{qin2024mobilenetv4,
      title={MobileNetV4 -- Universal Models for the Mobile Ecosystem}, 
      author={Danfeng Qin and Chas Leichner and Manolis Delakis and Marco Fornoni and Shixin Luo and Fan Yang and Weijun Wang and Colby Banbury and Chengxi Ye and Berkin Akin and Vaibhav Aggarwal and Tenghui Zhu and Daniele Moro and Andrew Howard},
      year={2024},
      eprint={2404.10518},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
