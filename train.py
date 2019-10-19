from net import resnext101_32x8d
import torch

model = resnext101_32x8d(num_classes=29)
image = torch.randn(size=(1, 3, 224, 224))
print(model(image).shape)