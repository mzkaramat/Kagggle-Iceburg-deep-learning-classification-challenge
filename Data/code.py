
# coding: utf-8

# In[154]:

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,TensorDataset
from sklearn.utils import shuffle
from torch.autograd import Variable
import torch.optim as optim

import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

#1604 enteries
train = pd.read_json('./trrain/data/processed/train.json')
train = shuffle(train)
# angle data
angles = np.array(train['inc_angle']);
# labels (iceberg = 1, ship = 0)
targets = np.array(train['is_iceberg']);

band_1 = np.concatenate([im for im in train['band_1']]).reshape(-1, 75, 75)
band_2 = np.concatenate([im for im in train['band_2']]).reshape(-1, 75, 75)
all_images = np.stack([band_1, band_2], axis=1)  # (1604, 2, 75, 75)
# all_images = [band_1, band_2], axis=3) # (1604, 75, 75, 2)

use_cuda = torch.cuda.is_available();


# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
# Tensor = FloatTensor

class IceBergDataset(Dataset):
    def __init__(self, dataset, offset, length):
        self.dataset = dataset
        self.offset = offset
        self.length = length
        assert len(dataset) >= offset + length, Exception("Parent Dataset not long enough")
        super(IceBergDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        assert i < self.length, Exception("index out of range")
        return self.dataset[i + self.offset]


def trainTestSplit(dataset, val_share=0.33):
    val_offset = int(len(dataset) * (1 - val_share))
    return IceBergDataset(dataset, 0, val_offset), IceBergDataset(dataset,
                                                                  val_offset, len(dataset) - val_offset)


def tensorConvertData(data):
    if (use_cuda):
        tensor = (torch.from_numpy(data).type(torch.FloatTensor).cuda())
    else:
        tensor = (torch.from_numpy(data)).type(torch.FloatTensor)
    return tensor


images = tensorConvertData(all_images);
targets = targets.reshape((targets.shape[0], 1))
targets = tensorConvertData(targets);
dataset = TensorDataset(images, targets);
# 1074, 530
dataset_train, dataset_val = trainTestSplit(dataset)

batch_size = 10
train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=1)
val_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1)


# models

class ConvNet1(nn.Module):
    def __init__(self):
        super(ConvNet1, self).__init__()
        self.conv1 = nn.Conv2d(2, 10, 15)  # torch.Size([batchsize, 10, 61, 61])
        self.conv2 = nn.Conv2d(10, 10, 15)  # torch.Size([batchsize, 10, 61, 47])
        self.conv3 = nn.Conv2d(10, 10, 15)  # torch.Size([batchsize, 10, 61, 33])
        self.pool = nn.MaxPool2d(2, 2)  # torch.Size([10, 10, 16, 16])
        self.fc1 = nn.Linear(10 * 16 * 16, 120)
        self.fc2 = nn.Linear(120, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 10 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.sig(self.fc2(x))
        return x


model = ConvNet1()
loss_func = torch.nn.BCELoss()
LR = 0.0005
MOMENTUM = 0.95
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-5)

for i, data in enumerate(train_loader):
    # just for 1 batch
    img, label = data
    if use_cuda:
        img, label = Variable(img.cuda(async=True)), Variable(label.cuda(async=True))
    else:
        img, label = Variable(img), Variable(label)

    output = model.forward(img)

    loss = loss_func(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("--" * 8)
    print(i)
    print(loss.data[0])

# In[ ]:




# In[ ]:


