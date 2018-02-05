
# coding: utf-8

# In[39]:


import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,TensorDataset
from sklearn.utils import shuffle
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import os





#1604 enteries
train = pd.read_json('./inputs/train.json')
train = shuffle(train)
# angle data
angles  = np.array(train['inc_angle']);
#labels (iceberg = 1, ship = 0)
targets = np.array(train['is_iceberg']);


band_1 = np.concatenate([im for im in train['band_1']]).reshape(-1, 75, 75)
band_2 = np.concatenate([im for im in train['band_2']]).reshape(-1, 75, 75)
all_images = np.stack([band_1, band_2], axis=1) # (1604, 2, 75, 75) 
#all_images = [band_1, band_2], axis=3) # (1604, 75, 75, 2)


use_cuda = torch.cuda.is_available();
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
# Tensor = FloatTensor

class IceBergDataset(Dataset):
    def __init__(self, dataset, offset, length):
        self.dataset = dataset
        self.offset = offset
        self.length = length
        assert len(dataset) >= offset+length, Exception("Parent Dataset not long enough")
        super(IceBergDataset, self).__init__()
    
    def __len__(self):        
        return self.length
    
    def __getitem__(self, i):
        assert i < self.length, Exception("index out of range")
        return self.dataset[i+self.offset]

def trainTestSplit(dataset, val_share=0.20):
    val_offset = int(len(dataset)*(1-val_share))
    return IceBergDataset(dataset, 0, val_offset), IceBergDataset(dataset, 
                                                  val_offset, len(dataset)-val_offset)
     

def tensorConvertData(data):
    if (use_cuda):
        tensor = (torch.from_numpy(data).type(torch.FloatTensor).cuda())    
    else:
        tensor = (torch.from_numpy(data)).type(torch.FloatTensor)     
    return tensor

#keep test set seperate 150 samples
test_images = all_images[1454:]
test_targets = targets[1454:]

#train and val set 1454 samples (1163 train, 291 val)
all_images =all_images[0:1454]
targets = targets[0:1454] 

#models

class ConvNet1(nn.Module):
    def __init__(self):
        super(ConvNet1, self).__init__()
        self.conv1 = nn.Conv2d(2, 10, 15) # torch.Size([batchsize, 10, 61, 61])
        self.conv2 = nn.Conv2d(10, 10, 15) # torch.Size([batchsize, 10, 61, 47])
        self.conv3 = nn.Conv2d(10, 10, 15) # torch.Size([batchsize, 10, 61, 33])
        self.pool = nn.MaxPool2d(2, 2)# torch.Size([10, 10, 16, 16])
        self.fc1 = nn.Linear(10 * 16 * 16, 120)
        self.fc2 = nn.Linear(120, 1)
        self.sig = nn.Sigmoid()

        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 10 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.sig(self.fc2(x))
        return x


#Get Data    
images = tensorConvertData(all_images);
targets=targets.reshape((targets.shape[0],1))
targets = tensorConvertData(targets);
dataset = TensorDataset(images, targets);
#1074, 530
dataset_train, dataset_val = trainTestSplit(dataset)

batch_size = 10
train_loader = DataLoader(dataset_train, batch_size=batch_size)
val_loader = DataLoader(dataset_val, batch_size=batch_size)


model = ConvNet1()
loss_func=torch.nn.BCELoss() 
rates = [0.001,0.0005,0.0001,0.00005,0.00001]

if use_cuda:
    model.cuda()
    loss_func.cuda()

#iterate over LR
for i in range(len(rates)):
    LR = rates[i]
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,weight_decay=5e-5)
    epochs = 650

    all_losses_training = []
    all_losses_validation = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(epochs):
    #train
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('*' * 5 + ':')
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
            
        all_losses_training.append(epoch_loss_train)
        print('Finish {} epoch, Training Loss: {:.6f}'.format(epoch + 1, epoch_loss_train))
        
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
            epoch_loss_val += loss.data[0] * label.size(0)
            
        all_losses_validation.append(epoch_loss_val)
        print('Finish {} epoch, Validation Loss: {:.6f}'.format(epoch + 1, epoch_loss_val))
    
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
        
    
    
    
    
    




