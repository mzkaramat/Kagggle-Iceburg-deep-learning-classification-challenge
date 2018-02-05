import pandas as pd
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


data_df = pd.read_csv('lat_features.csv', header=None)
#data_df = data_df.loc[:,data_df.sum(axis=0) !=0]
data = data_df.as_matrix()

tar = pd.read_csv('conv_images_tar.csv', header=None)
tar = tar.as_matrix()

def tensorConvertData(data):
    #if (use_cuda):
    #    tensor = (torch.from_numpy(data).type(torch.FloatTensor).cuda())
    #else:
    #    tensor = (torch.from_numpy(data)).type(torch.FloatTensor)
    tensor = (torch.from_numpy(data)).type(torch.FloatTensor)
    return tensor

def trainTestSplit(dataset, val_share=0.33):
    val_offset = int(len(dataset) * (1 - val_share))
    return IceBergDataset(dataset, 0, val_offset), IceBergDataset(dataset,
                                                                  val_offset, len(dataset) - val_offset)

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
    
    
num_epochs = 5
batch_size = 32
learning_rate = 1e-3
use_cuda = torch.cuda.is_available();


tar_t = tensorConvertData(tar);
data_t = tensorConvertData(data);
conv_images_dt  = TensorDataset(data_t, tar_t);

dataset_train, dataset_val = trainTestSplit(conv_images_dt)
train_loader = DataLoader(dataset_train, batch_size=batch_size)
val_loader = DataLoader(dataset_val, batch_size=batch_size)


class ClassNN(nn.Module):
    def __init__(self):
        super(ClassNN, self).__init__()
        self.fc1 = nn.Linear(44, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 5)
        self.fc5 = nn.Linear(5, 1)
        self.sig = nn.Sigmoid()

        
    def forward(self,x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.sig(x)
        return x
		
		
model = ClassNN()
loss_func=torch.nn.BCELoss() 
rates = [0.01]

if use_cuda:
    model.cuda()
    loss_func.cuda()

#iterate over LR
for i in range(len(rates)):
    LR = rates[i]
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,weight_decay=5e-5)
    epochs = 1

    all_losses_training = []
    all_losses_validation = []
    train_accuracies = []
    val_accuracies = []
    epoch_loss_train = 0.0
    epoch_loss_val = 0.0
    
    for epoch in range(epochs):
    #train
        epoch_loss_train = 0.0
        epoch_loss_val = 0.0
        epoch_acc_train = 0.0
        epoch_acc_val = 0.0
        
        model.train(True)
    
        #iterate over all batches
        for i, data in enumerate(train_loader,1):
            #get 1 batch
            img, label = data
            if use_cuda:
                img, label = Variable(img.cuda(async=True)), Variable(label.cuda(async=True))
            else:
                img, label = Variable(img), Variable(label)

            output = model.forward(img)
            
            #get loss
            loss = loss_func(output, label)
            epoch_loss_train += loss.data[0] * label.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.data[0] 
        epoch_loss_train = epoch_loss_train/len(train_loader)
        all_losses_training.append(epoch_loss_train)
        model.train(False)
        
        for i, data in enumerate(val_loader,1):
            img, label = data
            if use_cuda:
                    img = Variable(img.cuda(async=True), volatile=True)
                    label = Variable(label.cuda(async=True), volatile=True)
            else:
                img = Variable(img, volatile=True)
                label = Variable(label, volatile=True)

            out = model(img)

            loss = loss_func(out, label)
            epoch_loss_val += loss.data[0]
        epoch_loss_val = epoch_loss_val/len(val_loader)
        all_losses_validation.append(epoch_loss_val)
    
    plt.clf()
    x=np.arange(1,epochs+1)
    fig, ax = plt.subplots()
    ax.plot(x, all_losses_training,"r", label='Train Loss')
    ax.plot(x, all_losses_validation,"b", label='Validation Loss')

    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 15,
        }
    
    text = "LR: "+str(LR)
    
    plt.text(0.75, 0.75,text,horizontalalignment='center',verticalalignment='center',transform = ax.transAxes,fontdict=font)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title('Loss over Epochs')

    legend = ax.legend(loc='upper center', shadow=True)


    save_dir = "Graphs/LearningRates"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(save_dir+'/TrainingValidationLoss'+str(LR)+'.png')
    plt.show()