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


from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    print("exception")
    pass

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
    #if (use_cuda):
    #    tensor = (torch.from_numpy(data).type(torch.FloatTensor).cuda())
    #else:
    #    tensor = (torch.from_numpy(data)).type(torch.FloatTensor)
    tensor = (torch.from_numpy(data)).type(torch.FloatTensor)
    return tensor


#1604 enteries
train = pd.read_json('train.json')
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

print(use_cuda)




images = tensorConvertData(all_images);
targets = targets.reshape((targets.shape[0], 1))
targets = tensorConvertData(targets);
dataset = TensorDataset(images, targets);
dataset_train, dataset_val = trainTestSplit(dataset)
dataloader = DataLoader(dataset_train, batch_size=batch_size, num_workers=1)

class AutoEncoder(nn.Module):
    
    def __init__(self, code_size):
        super().__init__()
        self.code_size = code_size
        
        # Encoder specification
        self.enc_cnn_1 = nn.Conv2d(2, 10, kernel_size=5)
        self.enc_cnn_2 = nn.Conv2d(10, 20, kernel_size=5)
        self.enc_linear_1 = nn.Linear(4500, 2000)
        self.enc_linear_2 = nn.Linear(2000, self.code_size)
        
        # Decoder specification
        self.dec_linear_1 = nn.Linear(self.code_size, 2000)
        self.dec_linear_2 = nn.Linear(2000, IMAGE_SIZE*2)
        
    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code
    
    def encode(self, images):
        code = F.relu(self.enc_cnn_1(images))
        code = F.max_pool2d(code, 2)
        code = F.relu(self.enc_cnn_2(code))
        code = F.max_pool2d(code, 2)
        code = code.view([images.size(0), -1])
        
        code = F.relu(self.enc_linear_1(code))
        code = F.relu(self.enc_linear_2(code))
        return code
    
    def decode(self, code):
        out = F.selu(self.dec_linear_1(code))
        out = F.sigmoid(self.dec_linear_2(out))
        out = out.view([code.size(0), 2, IMAGE_WIDTH, IMAGE_HEIGHT])
        return out
    

IMAGE_SIZE = 5625
IMAGE_WIDTH = IMAGE_HEIGHT = 75


code_size = 1000
num_epochs = 1
batch_size = 128
lr = 0.001
optimizer_cls = optim.Adam


# Instantiate model
autoencoder = AutoEncoder(code_size).cuda()
loss_fn = nn.MSELoss()
optimizer = optimizer_cls(autoencoder.parameters(), lr=lr)


# Training loop
for epoch in range(num_epochs):
    print("Epoch %d" % epoch)
      
    
    for data in dataloader:
        image, _ = data
        img = Variable(image.cuda())
        # ===================forward=====================
        out,code = autoencoder(img)
        optimizer.zero_grad()
        loss = loss_fn(out, img)
        loss.backward()
        optimizer.step()
        
    print("Loss = %.3f" % loss.data[0])	