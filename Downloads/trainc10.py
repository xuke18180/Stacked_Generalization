# -*- coding: utf-8 -*-
"""trainc10.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rt3rAaWGQxrwrLpujKVbvsx_2UBDB0_m
"""

!pip install awscli

!aws configure

!aws s3 cp s3://jmao-penn/inpca_models/big_run/ ./big_run/ --recursive

import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import torch as th
from fastcore.script import *
import scipy.sparse.linalg as sp
import seaborn as sns
import matplotlib.pyplot as plt

import h5py

yh_g = th.load('big_run/{"seed":42,"bseed":-1,"aug":"none","m":"allcnn","bn":true,"drop":0.0,"opt":"adam","bs":1000,"lr":0.005,"wd":1e-05}.p')

print(yh_g[66])

yh = [th.stack([x['yh'] for x in yh_g])]

yh.shape

yvh = [th.stack([x['yvh'] for x in yh_g])]

yvh.shape

yh_g = th.load('big_run/{"seed":42,"bseed":-1,"aug":"none","m":"allcnn","bn":true,"drop":0.0,"opt":"adam","bs":200,"lr":0.001,"wd":1e-05}.p')

yh.append(th.stack([x['yh'] for x in yh_g['data']]))
yvh.append(th.stack([x['yvh'] for x in yh_g['data']]))

yh_g = th.load('/content/big_run/{"seed":42,"bseed":-1,"aug":"none","m":"allcnn","bn":true,"drop":0.0,"opt":"adam","bs":200,"lr":0.0005,"wd":1e-05,"corner":"normal"}.p')

yh.append(th.stack([x['yh'] for x in yh_g['data']]))
yvh.append(th.stack([x['yvh'] for x in yh_g['data']]))

x = [x[7] for x in yh]


x = th.cat(x, dim=1)

print(x.shape)

x_val = [y[7] for y in yvh]
x_val = th.cat(x_val, dim=1)
print(x_val.shape)

print(x.shape)

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


transform = transforms.ToTensor()
cifar_train = CIFAR10(root='./data', train=True, download=True, transform=transform)

cifar_labels = torch.tensor(cifar_train.targets)

class CIFARCustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.labels[idx]
        return x, y


train_dataset = CIFARCustomDataset(x, cifar_labels)


train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

cifar_val = CIFAR10(root='./data', train=False, download=True, transform=transform)
cifar_val_labels = torch.tensor(cifar_val.targets)

val_dataset = CIFARCustomDataset(x_val, cifar_val_labels)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

import torch
import torch.nn as nn
import torchvision.models as models

class stackGen(nn.Module):
    def __init__(self):
        super(stackGen, self).__init__()
        self.fc = nn.Linear(30, 50)
        self.bn = nn.BatchNorm1d(50)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(0.5)
        self.fc1 = nn.Linear(50, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dp(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dp1(x)
        x = self.fc2(x)
        return x

stackGen = stackGen()
optim = torch.optim.Adam(stackGen.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 1000

def val(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


for epoch in range(epochs):
    stackGen.train()
    for i, (inputs, labels) in enumerate(train_loader):
        optim.zero_grad()
        outputs = stackGen(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

            val_acc = val(stackGen, val_loader)
            print(f'Epoch [{epoch+1}/{epochs}], Validation Accuracy: {val_acc:.2f}%')