%matplotlib inline
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
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os
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
num_epochs = 5
batch_size = 32
learning_rate = 1e-3
use_cuda = torch.cuda.is_available();


targets = np.array(train['is_iceberg']);
band_1 = np.concatenate([im for im in train['band_1']]).reshape(-1, 75, 75)
band_2 = np.concatenate([im for im in train['band_2']]).reshape(-1, 75, 75)
all_images = np.stack([band_1, band_2], axis=1)  # (1604, 2, 75, 75)

conv_images_set = all_images
conv_images_tar = targets
conv_images_set = tensorConvertData(conv_images_set);
conv_images_tar = conv_images_tar.reshape((conv_images_tar.shape[0], 1))
conv_images_tar = tensorConvertData(conv_images_tar);
conv_images_dt  = TensorDataset(conv_images_set, conv_images_tar);
conv_images_loader = DataLoader(conv_images_dt, batch_size=batch_size)



print(use_cuda)
#keep test set seperate 150 samples
test_images = all_images[1454:]
test_targets = targets[1454:]

test_images = tensorConvertData(test_images);
test_targets = test_targets.reshape((test_targets.shape[0], 1))
test_targets = tensorConvertData(test_targets);
dataset_test = TensorDataset(test_images, test_targets);



#train and val set 1454 samples (1163 train, 291 val)
all_images =all_images[0:1454]
targets = targets[0:1454] 

images = tensorConvertData(all_images);
targets = targets.reshape((targets.shape[0], 1))
targets = tensorConvertData(targets);
dataset = TensorDataset(images, targets);
dataset_train, dataset_val = trainTestSplit(dataset)

train_loader = DataLoader(dataset_train, batch_size=batch_size)
val_loader = DataLoader(dataset_val, batch_size=batch_size)
test_loader = DataLoader(dataset_test, batch_size=batch_size)

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
        self.dec_linear_2 = nn.Linear(2000, 4500)
        self.dec_cnn_1 =  nn.ConvTranspose2d(20, 20, kernel_size=5,stride=2,padding=1)
        self.dec_cnn_2 =  nn.ConvTranspose2d(20, 10, kernel_size=5)
        self.dec_cnn_3 = nn.ConvTranspose2d(10, 10, kernel_size=5,stride=2,padding=1)
        self.dec_cnn_4 = nn.ConvTranspose2d(10, 2, kernel_size=5)
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
        out = F.relu(self.dec_linear_1(code))
        out = F.relu(self.dec_linear_2(out))
        out = out.view([code.size(0), 20, 15, 15])
        out = F.relu(self.dec_cnn_1(out))
        out = F.relu(self.dec_cnn_2(out))
        out = F.relu(self.dec_cnn_3(out))
        out = self.dec_cnn_4(out)
        return out
    

IMAGE_SIZE = 5625
IMAGE_WIDTH = IMAGE_HEIGHT = 75

code_size = 100
num_epochs = 1000
lr = 0.0001
optimizer_cls = optim.Adam
rates = [0.0001]

# Instantiate model
model = AutoEncoder(code_size).cuda()
loss_fn = nn.MSELoss().cuda()



for i in range(len(rates)):
    lr = rates[i]
    optimizer = optimizer_cls(model.parameters(), lr=lr)
    all_losses_training = []
    all_losses_validation = []
    
    epoch_loss_train = 0.0
    epoch_loss_val = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        print("Epoch %d" % epoch)
        epoch_loss_train = 0.0
        epoch_loss_val = 0.0
        epoch_loss_val = 0.0
        for data in train_loader:
            image, _ = data
            img = Variable(image.cuda())
            # ===================forward=====================
            out,code = model(img)
            optimizer.zero_grad()
            loss = loss_fn(out, img)
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.data[0] 
        epoch_loss_train = epoch_loss_train/len(train_loader)
        print("Train Loss = %.3f" % epoch_loss_train)
        all_losses_training.append(epoch_loss_train)

        for i, data in enumerate(val_loader,1):
                image, _ = data
                img = Variable(image.cuda())

                out,code = model(img)

                loss = loss_fn(out, img)
                                
                epoch_loss_val += loss.data[0]
        epoch_loss_val = epoch_loss_val/len(val_loader)
        print("Test mse = %.3f" % epoch_loss_val)
        all_losses_validation.append(epoch_loss_val)

    plt.clf()
    x=np.arange(1,num_epochs+1)
    fig, ax = plt.subplots()
    ax.plot(x, all_losses_training,"r", label='Train Loss')
    ax.plot(x, all_losses_validation,"b", label='Validation Loss')

    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 15,
        }

    text = "LR: "+str(lr)

    plt.text(0.75, 0.75,text,horizontalalignment='center',verticalalignment='center',transform = ax.transAxes,fontdict=font)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title('Loss over Epochs')

    legend = ax.legend(loc='upper center', shadow=True)


    save_dir = "Graphs/LearningRates"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #plt.savefig(save_dir+'/TrainingValidationLoss'+str(lr)+'.png')
    plt.show()
torch.save(model.state_dict(), './conv_autoencoder.pth')

lat_features = []
for i, data in enumerate(conv_images_loader,1):
    image, _ = data
    img = Variable(image.cuda())
    out,code = model(img)
    
    if(i==1):
        lat_features = code
    else:
        lat_features = torch.cat((lat_features,code),0)
    
    loss = loss_fn(out, img)
print("Test mse = %.3f" % loss.data[0])

lat_features = []
for i, data in enumerate(conv_images_loader,1):
    image, _ = data
    img = Variable(image.cuda())
    out,code = model(img)
    
    if(i==1):
        lat_features = code
    else:
        lat_features = torch.cat((lat_features,code),0)
    
    loss = loss_fn(out, img)
    print(out)
    print(img)
print("Test mse = %.3f" % loss.data[0])

temp = lat_features.cpu()
conv_features = temp.data.numpy()
conv_features.shape
np.savetxt("lat_features_1.csv", conv_features, delimiter=",")