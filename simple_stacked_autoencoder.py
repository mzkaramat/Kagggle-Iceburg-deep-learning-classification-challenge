import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,TensorDataset
from sklearn.utils import shuffle
from torch.autograd import Variable
import torch.optim as optim
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

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


#1604 enteries
train = pd.read_json('./Data/trrain/data/processed/train.json')
train = shuffle(train)
# angle data
angles = np.array(train['inc_angle']);
# labels (iceberg = 1, ship = 0)
targets = np.array(train['is_iceberg']);
band_1 = np.concatenate([im for im in train['band_1']]).reshape(-1, 75, 75)
band_2 = np.concatenate([im for im in train['band_2']]).reshape(-1, 75, 75)
all_images = np.stack([band_1, band_2], axis=1)  # (1604, 2, 75, 75)

all_images.shape
num_epochs = 5
batch_size = 5
learning_rate = 1e-3

use_cuda = torch.cuda.is_available();

images = tensorConvertData(all_images);
targets = targets.reshape((targets.shape[0], 1))
targets = tensorConvertData(targets);
dataset = TensorDataset(images, targets);
dataset_train, dataset_val = trainTestSplit(dataset)

dataloader = DataLoader(dataset_train, batch_size=batch_size, num_workers=1)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(75 * 75, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 75 * 75), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

		
model = autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    print("epoch "+str(epoch))
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img)
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data[0]))
    