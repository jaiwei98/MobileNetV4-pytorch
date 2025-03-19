import timm
import argparse
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from mobilenet.mobilenetv4 import MobileNetV4

TRANSFORMS_CONFIG = {
    'input_size': (3, 224, 224), 
    'interpolation': 'bicubic', 
    'mean': (0.485, 0.456, 0.406), 
    'std': (0.229, 0.224, 0.225), 
    'crop_pct': 0.875, # 0.95
    'crop_mode': 'center'
}

class MyModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = MobileNetV4(model_name)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1280, 1000, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        return self.linear(self.flatten(x[-1]))

def preprocess_test_image(image_path):
    transforms_ = timm.data.create_transform(**TRANSFORMS_CONFIG, is_training=False)
    img = Image.open(image_path).convert("RGB")
    return transforms_(img).unsqueeze(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="MobileNetV4ConvSmall", help="Insert from this options (MobileNetV4ConvSmall, MobileNetV4ConvMedium, MobileNetV4ConvLarge)")
    parser.add_argument("--weight-path", type=str, default="")
    parser.add_argument("--image-path", type=str, default="scripts/sample.jpg")

    args = parser.parse_args()
    
    # Load model
    model = MyModel(args.model_name)
    model.load_state_dict(torch.load(args.weight_path, weights_only=True))
    model.eval()

    # Process image and infer
    x = preprocess_test_image(args.image_path)
    y = model(x)
    probs = F.softmax(y, dim=1)
    print(torch.argmax(probs, dim=1))

if __name__ == "__main__":
    main()