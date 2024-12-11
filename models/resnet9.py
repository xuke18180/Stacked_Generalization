import torch as ch
import torch.nn as nn

class Mul(ch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def forward(self, x):
        return x * self.weight

class Flatten(ch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Residual(ch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, x):
        return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return ch.nn.Sequential(
        ch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                     stride=stride, padding=padding, groups=groups, bias=False),
        ch.nn.BatchNorm2d(channels_out),
        ch.nn.ReLU(inplace=True)
    )

class ResNet9(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.model = ch.nn.Sequential(
            conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
            conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
            Residual(ch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
            conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
            ch.nn.MaxPool2d(2),
            Residual(ch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
            conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
            ch.nn.AdaptiveMaxPool2d((1, 1)),
            Flatten(),
            ch.nn.Linear(128, num_classes, bias=False),
            Mul(0.2)
        )
        
    def forward(self, x: ch.Tensor) -> ch.Tensor:
        return self.model(x)  # Return logits directly, just like PyTorch models

