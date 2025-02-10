import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset, DataLoader, ConcatDataset
from torchvision.models import ResNet18_Weights
from torch import optim
from PIL import Image
import torch.optim.lr_scheduler as lr_scheduler
from scipy import ndimage
import nibabel as nib
import pandas as pd
import numpy as np
import os
import re
import time
from random import sample
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

image_dir = "/u/project/mscohen/hwang/data/bcell/dataset2-master/dataset2-master/images"
train_dir = "TRAIN"
val_dir = "TEST_SIMPLE"
test_dir = "TEST"
train_path = os.path.join(image_dir, train_dir)
val_path = os.path.join(image_dir, val_dir)
test_path = os.path.join(image_dir, test_dir)
cell_class = os.listdir(os.path.join(image_dir, train_dir))
print(cell_class)

def transform_data(dataset):
    if dataset == "TRAIN":
        
        transform = transforms.Compose([
        #transforms.Resize(size=(224, 224)), 
        #transforms.RandomHorizontalFlip(), 
        #transforms.RandomRotation(degrees=15), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    else:
        transform = transforms.Compose([
        #transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

dataset_train = datasets.ImageFolder(train_path, transform=transform_data("TRAIN"))
train_loader = DataLoader(dataset_train, batch_size=128, shuffle=True) 
##test dataset loader
#imgs, labels = next(train_loader.__iter__())
#print(imgs.shape, labels)

dataset_val = datasets.ImageFolder(val_path, transform=transform_data("TEST_SIMPLE"))
val_loader = DataLoader(dataset_val, batch_size=128, shuffle=True)

dataset_test = datasets.ImageFolder(test_path, transform=transform_data("TEST"))
test_loader = DataLoader(dataset_test, batch_size=128, shuffle=True) 

data_loader = {}
data_loader["train"] = train_loader
data_loader["val"] = val_loader
data_loader["test"] = test_loader

data_len = {}
data_len["train"] = len(train_loader)
data_len["val"] = len(val_loader)
data_len["test"] = len(test_loader)
print("train_len, val_len, test_len", data_len["train"], data_len["val"], data_len["test"])

class model_vgg(nn.Module):
    def __init__(self):
        super().__init__()

        ##input [32, 3, 240, 320]
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding="same")
        self.norm1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding="same")
        self.norm2 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding="same")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding="same")
        self.norm3 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding="same")
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding="same")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding="same")
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding="same")
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding="same")
        self.norm4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        

        self.fc1 = nn.Linear(in_features=512*15*20, out_features=4096)
        #self.fc1 = nn.Linear(in_features=256*30*40, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=1000)
        self.fc3 = nn.Linear(in_features=1000, out_features=4)
        
    def forward(self, x):
        x = F.relu(self.norm1(self.conv1_1(x)))
        x = F.relu(self.norm1(self.conv1_2(x)))
        x = self.pool1(x)
        x = F.relu(self.norm2(self.conv2_1(x)))
        x = F.relu(self.norm2(self.conv2_2(x)))
        x = self.pool2(x)
        x = F.relu(self.norm3(self.conv3_1(x)))
        x = F.relu(self.norm3(self.conv3_2(x)))
        x = F.relu(self.norm3(self.conv3_3(x)))
        x = self.pool3(x)
        x = F.relu(self.norm4(self.conv4_1(x)))
        x = F.relu(self.norm4(self.conv4_2(x)))
        x = F.relu(self.norm4(self.conv4_3(x)))
        x = self.pool4(x)
        #print(x.shape)
        x = x.view(-1, 512*15*20)
        #x = x.view(-1, 256*30*40)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class model_resnet18(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding="same")
        self.norm0 = nn.BatchNorm2d(64)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ##first block does not have stride=2, the channels stay as 64, so no need to downsample
        self.conv1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        ##first layer with stride=2 and the rest with stride=1
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(128)
        ##use stride=2 and kernel=1 to generate same 128 channels with 1/2*W and 1/2*H
        self.downsample2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm2d(256)
        self.downsample3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.norm4 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.downsample4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, padding=0)
        
        #self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=1, stride=stride, padding=0), 
        #                                          nn.BatchNorm2d(out_chan))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out_conv0 = F.relu(self.norm0(self.conv0(x)))
        out_pool0 = self.pool0(out_conv0)
        print("pool0", out_pool0.shape)

        out_conv1_1 = F.relu(self.norm1(self.conv1_1(out_pool0)))
        out_conv1_2 = self.norm1(self.conv1_2(out_conv1_1))
        print("conv1_2", out_conv1_2.shape)
        out_conv1 = F.relu(torch.add(out_conv1_2, out_pool0))
        print("out_conv1", out_conv1.shape)

        out_conv1_3 = F.relu(self.norm1(self.conv1_1(out_conv1)))
        out_conv1_4 = self.norm1(self.conv1_2(out_conv1_3))
        print("conv1_4", out_conv1_4.shape)
        out_conv1 = F.relu(torch.add(out_conv1_4, out_conv1))
        print("out_conv1", out_conv1.shape)
        
        out_conv2_1 = F.relu(self.norm2(self.conv2_1(out_conv1)))
        out_conv2_2 = self.norm2(self.conv2_2(out_conv2_1))
        print("out_conv2_2", out_conv2_2.shape)
        out_conv2 =  F.relu(torch.add(out_conv2_1, self.downsample2(out_conv1)))
        print("out_conv2", out_conv2.shape)

        out_conv2_3 = F.relu(self.norm2(self.conv2_3(out_conv2)))
        out_conv2_4 = self.norm2(self.conv2_4(out_conv2_3))
        print("out_conv2_4", out_conv2_4.shape)
        out_conv2 =  F.relu(torch.add(out_conv2_4, out_conv2))
        print("out_conv_2", out_conv2.shape)

        out_conv3_1 = F.relu(self.norm3(self.conv3_1(out_conv2)))
        out_conv3_2 = self.norm3(self.conv3_2(out_conv3_1))
        print("out_conv3_2", out_conv3_2.shape)
        out_conv3 = F.relu(torch.add(out_conv3_2, self.downsample3(out_conv2)))
        print("out_conv3", out_conv3.shape)

        out_conv3_3 = F.relu(self.norm3(self.conv3_3(out_conv3)))
        out_conv3_4 = self.norm3(self.conv3_4(out_conv3_3))
        print("out_conv3_4", out_conv3_4.shape)
        out_conv3 = F.relu(torch.add(out_conv3_4, out_conv3))
        print("out_conv3", out_conv3.shape)        

        out_conv4_1 = F.relu(self.norm4(self.conv4_1(out_conv3)))
        out_conv4_2 = self.norm4(self.conv4_2(out_conv4_1))
        print("out_conv4_2", out_conv4_2.shape)
        out_conv4 = F.relu(torch.add(out_conv4_2, self.downsample4(out_conv3)))
        print("out_conv4", out_conv4.shape)

        out_conv4_3 = F.relu(self.norm4(self.conv4_3(out_conv4)))
        out_conv4_4 = self.norm4(self.conv4_4(out_conv4_3))
        out_conv4 = F.relu(torch.add(out_conv4_4, out_conv4))
        print("out_conv4", out_conv4.shape)
        
        out = self.avgpool(out_conv4)
        print("avelayer", out.shape)
        out = out.view(out.size(0), -1)
        print("view", out.shape)
        out = self.fc(out)
        print(out.shape)
        return out



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#model = model_vgg().to(device)
model = model_vgg_resnet().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

losses = []
accuracies = []
val_losses = []
val_accuracies = []
EPOCHS = 30 
print("Total Epochs:", EPOCHS)
start = time.time()

for epoch in range(EPOCHS):
    epoch_loss = 0
    epoch_accuracy = 0
    for phase in ['train', 'val']:
        predictions, targets = [], []
        for i, data in enumerate(data_loader[phase], 0):
            X, y = data
            #print("data loader No. ", i)
            #print("y shape ", y.shape)
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            predict = model(X)
            loss = loss_fn(predict, y)
            #print(predict)
            #print("y ", y)
            #print("loss", loss.item())
            # forward + backward + optimize
            with torch.set_grad_enabled(phase == 'train'):
                loss.backward()
                #for name, param in model.named_parameters():
                #    if name == "conv3_3.weight":
                #        print("before: ", name, param.grad)
                #update weights
                optimizer.step()
                #for name, param in model.named_parameters():
                #    if name == "conv3_3.weight":
                #        print("after ", name, param.grad)
            ##Loss
            epoch_loss += loss
            ##accuracy
            predict_max = predict.argmax(dim=1).to(torch.float)
            #print("predict_argmax", predict_max)
            accuracy = (predict_max == y).float().mean()
            #print("accuracy", accuracy)
            epoch_accuracy += accuracy
            ##for score
            prediction = predict_max.cpu().numpy()
            predictions.extend(prediction)
            target = y.cpu().numpy()
            targets.extend(target)

        epoch_loss = epoch_loss / data_len[phase]
        if phase == 'val':
            print("acc len: ", epoch_accuracy, data_len[phase])
        epoch_accuracy = epoch_accuracy / data_len[phase]
        acc_score = accuracy_score(targets, predictions)
        f1_macro = f1_score(targets, predictions, average='macro')
        print("Epoch: {}, phase: {}, loss: {:.4f}, accracy: {:.4f}, acc_score: {:.4f}, f1: {:.4f}, time: {}".format(epoch, phase, epoch_loss, epoch_accuracy, acc_score, f1_macro, time.time() - start))

checkpoint_path = "resnet_checkpoint_bcell.pth"
torch.save(model.state_dict(), checkpoint_path)

phase="test"
with torch.no_grad():
    predictions, targets = [],[]
    for i, data in enumerate(data_loader[phase], 0):
            X, y = data
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            predict = model(X)
            predict_max = predict.argmax(dim=1).to(torch.float)
            loss = loss_fn(predict, y)
            epoch_loss += loss

            accuracy = (predict_max == y).float().mean()
            epoch_accuracy += accuracy

            prediction = predict_max.cpu().numpy()
            predictions.extend(prediction)
            target = y.cpu().numpy()
            targets.extend(target)

    epoch_loss = epoch_loss / data_len[phase]
    epoch_accuracy = epoch_accuracy / data_len[phase]
    acc_score = accuracy_score(targets, predictions)
    f1_macro = f1_score(targets, predictions, average='macro')
    print("phase: {}, loss: {:.4f}, accracy: {:.4f}, acc_score: {:.4f}, f1: {:.4f}, time: {}".format(phase, epoch_loss, epoch_accuracy, acc_score, f1_macro, time.time() - start))
