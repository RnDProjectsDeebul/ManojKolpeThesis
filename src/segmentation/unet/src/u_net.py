import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle

from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print("Device:", device)

train_dir_color = '/scratch/mkolpe2s/MT/Main_data_folder/Segmentation_dataset/scannet/experiment_data/data/color_train/'
train_dir_label = '/scratch/mkolpe2s/MT/Main_data_folder/Segmentation_dataset/scannet/experiment_data/data/label_train/'
val_dir_color = '/scratch/mkolpe2s/MT/Main_data_folder/Segmentation_dataset/scannet/experiment_data/data/color_valid/'
val_dir_label = '/scratch/mkolpe2s/MT/Main_data_folder/Segmentation_dataset/scannet/experiment_data/data/label_valid/'

train_fns_color = os.listdir(train_dir_color)
train_fns_label = os.listdir(train_dir_label)
val_fns_color = os.listdir(val_dir_color)
val_fns_label = os.listdir(val_dir_label)

print("train_fns_color:", len(train_fns_color), "train_fns_label:",len(train_fns_label),"val_fns_color:", len(val_fns_color),"val_fns_label:", len(val_fns_label))

num_items = 1000
color_array = np.random.choice(range(256), 3*num_items).reshape(-1, 3)

num_classes = 40
label_model = KMeans(n_clusters=num_classes)
label_model.fit(color_array)

class scannet(Dataset):
    
    def __init__(self, image_dir, label_dir, label_model):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_fns = os.listdir(image_dir)
        self.label_fns = os.listdir(label_dir)
        self.label_model = label_model
        
    def __len__(self):
        return len(self.image_fns)
    
    def __getitem__(self, index):
        image_fn = self.image_fns[index]
        label_fn = self.label_fns[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        image = Image.open(image_fp).convert('RGB')
        image = np.array(image)
        label_fp = os.path.join(self.label_dir, image_fn)
        label = Image.open(label_fp).convert('RGB')
        label = np.array(label)
        cityscape, label = image, label
        label_class = self.label_model.predict(label.reshape(-1, 3)).reshape(240, 320)
        cityscape = self.transform(cityscape)
        label_class = torch.Tensor(label_class).long()
        return cityscape, label_class, 
    
    def split_image(self, image):
        image = np.array(image)
        cityscape, label = image[:, :256, :], image[:, 256:, :]
        return cityscape, label
    
    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform_ops(image)

dataset = scannet(train_dir_color, train_dir_label, label_model)
print("Length of dataset:", len(dataset))

class UNet(nn.Module):
    
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.contracting_11 = self.conv_block(in_channels=3, out_channels=64)
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_31 = self.conv_block(in_channels=128, out_channels=256)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_41 = self.conv_block(in_channels=256, out_channels=512)
        self.contracting_42 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = self.conv_block(in_channels=512, out_channels=1024)
        self.expansive_11 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_12 = self.conv_block(in_channels=1024, out_channels=512)
        self.expansive_21 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_22 = self.conv_block(in_channels=512, out_channels=256)
        self.expansive_31 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=256, out_channels=128)
        self.expansive_41 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_42 = self.conv_block(in_channels=128, out_channels=64)
        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)

        self.gamma2 = nn.Parameter(torch.randn(1), requires_grad=True).float()
        self.ell = nn.Parameter(torch.randn(1), requires_grad=True).float()
        self.sigma2 = nn.Parameter(torch.randn(1), requires_grad=True).float()
        
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels))
        return block
    
    def forward(self, X):
        contracting_11_out = self.contracting_11(X) # [-1, 64, 256, 256]
        contracting_12_out = self.contracting_12(contracting_11_out) # [-1, 64, 128, 128]
        contracting_21_out = self.contracting_21(contracting_12_out) # [-1, 128, 128, 128]
        contracting_22_out = self.contracting_22(contracting_21_out) # [-1, 128, 64, 64]
        contracting_31_out = self.contracting_31(contracting_22_out) # [-1, 256, 64, 64]
        contracting_32_out = self.contracting_32(contracting_31_out) # [-1, 256, 32, 32]
        contracting_41_out = self.contracting_41(contracting_32_out) # [-1, 512, 32, 32]
        contracting_42_out = self.contracting_42(contracting_41_out) # [-1, 512, 16, 16]
        middle_out = self.middle(contracting_42_out) # [-1, 1024, 16, 16]
        
        # with open('middle_out_scannet.pkl', 'wb') as fp:
        #     pickle.dump(middle_out, fp)
        # D = torch.from_numpy(np.expand_dims(genDistM(poses), 0))
        # middle_out = self.gplayer(D, middle_out)
        
        # with open('/content/drive/MyDrive/Master_thesis/test_result/D.pkl', 'rb') as fp:
        #       D = pickle.load(fp)
    
        # D = torch.from_numpy(np.expand_dims(D, 0))
        # l = 1
        
        # gamma2 = nn.Parameter(torch.randn(1), requires_grad=True).float()
        # ell = nn.Parameter(torch.randn(1), requires_grad=True).float()
        # sigma2 = nn.Parameter(torch.randn(1), requires_grad=True).float()

        # print(middle_out.shape)

        # a, b, c, d = middle_out.size()
        # print(a, b, c, d)
        # Y = middle_out.view(1,-1).cpu().float()
        # print(Y.shape)
        # D = D.float()
        # K = torch.exp(gamma2) * (1 + math.sqrt(3) * D / torch.exp(ell)) * torch.exp(-math.sqrt(3) * D / torch.exp(ell))
        # I = torch.eye(l).expand(l, l).float()
        # X,_ = torch.solve(Y, K+torch.exp(sigma2)*I)
        # Z = K.bmm(X)
        # Z = F.relu(Z)
        # middle_out = Z[:, 0].view(a, 1024, 15, 20).to(device)

        expansive_11_out = self.expansive_11(middle_out) # [-1, 512, 32, 32]
        expansive_12_out = self.expansive_12(torch.cat((expansive_11_out, contracting_41_out), dim=1)) # [-1, 1024, 32, 32] -> [-1, 512, 32, 32]
        expansive_21_out = self.expansive_21(expansive_12_out) # [-1, 256, 64, 64]
        expansive_22_out = self.expansive_22(torch.cat((expansive_21_out, contracting_31_out), dim=1)) # [-1, 512, 64, 64] -> [-1, 256, 64, 64]
        expansive_31_out = self.expansive_31(expansive_22_out) # [-1, 128, 128, 128]
        expansive_32_out = self.expansive_32(torch.cat((expansive_31_out, contracting_21_out), dim=1)) # [-1, 256, 128, 128] -> [-1, 128, 128, 128]
        expansive_41_out = self.expansive_41(expansive_32_out) # [-1, 64, 256, 256]
        expansive_42_out = self.expansive_42(torch.cat((expansive_41_out, contracting_11_out), dim=1)) # [-1, 128, 256, 256] -> [-1, 64, 256, 256]
        output_out = self.output(expansive_42_out) # [-1, num_classes, 256, 256]
        return output_out

# model = UNet(num_classes=num_classes)

batch_size = 16

epochs = 150
lr = 0.001

dataset = scannet(train_dir_color, train_dir_label, label_model)
data_loader = DataLoader(dataset, batch_size=batch_size)

with open('/scratch/mkolpe2s/MT/code/result/data_loader.pkl', 'wb') as fp:
    pickle.dump(data_loader, fp)

model = UNet(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

step_losses = []
epoch_losses = []
for epoch in tqdm(range(epochs)):
    epoch_loss = 0
    for X, Y in tqdm(data_loader, total=len(data_loader), leave=False):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        Y_pred = model(X)

        # Y = Y.cpu().detach().numpy()
        # Y_pred = Y_pred.cpu().detach().numpy()

        # with open('/scratch/mkolpe2s/MT/code/result_v2/Y_o.pkl', 'wb') as fp:
        #     pickle.dump(Y, fp)
        # with open('/scratch/mkolpe2s/MT/code/result_v2/Y_pred_o.pkl', 'wb') as fp:
        #     pickle.dump(Y_pred, fp)
        # sys.exit("Error message")
        loss = criterion(Y_pred, Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        step_losses.append(loss.item())
        print("epoch_loss:",epoch_loss)
        print("step_losses:",loss.item())
    epoch_losses.append(epoch_loss/len(data_loader))

with open('/scratch/mkolpe2s/MT/code/result/epoch_losses.pkl', 'wb') as fp:
    pickle.dump(epoch_losses, fp)

with open('/scratch/mkolpe2s/MT/code/result/step_losses.pkl', 'wb') as fp:
    pickle.dump(step_losses, fp)

model_name = "/scratch/mkolpe2s/MT/code/result/U-Net.pth"
torch.save(model.state_dict(), model_name)

class scannet(Dataset):
    
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_fns = os.listdir(image_dir)
        self.label_fns = os.listdir(label_dir)
        
    def __len__(self):
        return len(self.image_fns)
    
    def __getitem__(self, index):
        image_fn = self.image_fns[index]
        label_fn = self.label_fns[index]
        print(index)
        image_fp = os.path.join(self.image_dir, image_fn)
        image = Image.open(image_fp).convert('RGB')
        image = np.array(image)
        print(image_fp)
        label_fp = os.path.join(self.label_dir, image_fn)
        print(label_fp)
        label = Image.open(label_fp)
        label = np.array(label)
        print(label.shape)
        print("----------------------------------------")
        cityscape, label = image, label
        cityscape = self.transform(cityscape)
        label_class = torch.Tensor(label).long()
        return cityscape, label_class, 
    
    def split_image(self, image):
        image = np.array(image)
        cityscape, label = image[:, :256, :], image[:, 256:, :]
        return cityscape, label
    
    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform_ops(image)

test_batch_size = 8
dataset = scannet(val_dir_color, val_dir_label)
data_loader = DataLoader(dataset, batch_size=test_batch_size)

X, Y = next(iter(data_loader))
X, Y = X.to(device), Y.to(device)
Y_pred = model(X)
Y_pred = torch.argmax(Y_pred, dim=1)

inverse_transform = transforms.Compose([
    transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
])

X = X.cpu().detach().numpy()
Y = Y.cpu().detach().numpy()
Y_pred = Y_pred.cpu().detach().numpy()

print(X.shape, Y.shape, Y_pred.shape)

with open('/scratch/mkolpe2s/MT/code/result/X.pkl', 'wb') as fp:
    pickle.dump(X, fp)

with open('/scratch/mkolpe2s/MT/code/result/Y.pkl', 'wb') as fp:
    pickle.dump(Y, fp)

with open('/scratch/mkolpe2s/MT/code/result/Y_pred.pkl', 'wb') as fp:
    pickle.dump(Y_pred, fp)