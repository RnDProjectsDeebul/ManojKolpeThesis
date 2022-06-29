import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle
import math
import sys
from numpy.linalg import inv

from tqdm import tqdm
import telepot
bot = telepot.Bot('5297815301:AAF6dr1AfH4BGPu1DeJ0vE2vhddA3bMTkEg')

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print("Device:", device)

train_dir_color = '/scratch/mkolpe2s/MT/Main_data_folder/Segmentation_dataset/scannet/experiment_data/data/color_train/'
train_dir_label = '/scratch/mkolpe2s/MT/Main_data_folder/Segmentation_dataset/scannet/experiment_data/data/label_train_n/'
val_dir_color = '/scratch/mkolpe2s/MT/Main_data_folder/Segmentation_dataset/scannet/experiment_data/data/color_valid/'
val_dir_label = '/scratch/mkolpe2s/MT/Main_data_folder/Segmentation_dataset/scannet/experiment_data/data/label_valid_n/'
train_dir_pose = '/scratch/mkolpe2s/MT/Main_data_folder/Segmentation_dataset/scannet/experiment_data/data/pose_train/'
val_dir_pose = '/scratch/mkolpe2s/MT/Main_data_folder/Segmentation_dataset/scannet/experiment_data/data/pose_valid/'

train_fns_color = os.listdir(train_dir_color)
train_fns_label = os.listdir(train_dir_label)
val_fns_color = os.listdir(val_dir_color)
val_fns_label = os.listdir(val_dir_label)

print("train_fns_color:", len(train_fns_color), "train_fns_label:",len(train_fns_label),"val_fns_color:", len(val_fns_color),"val_fns_label:", len(val_fns_label))

def genDistM(poses):
    n = len(poses)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = pose_distance(poses[i], poses[j])
    return D


def pose_distance(p1, p2):
    rel_pose = np.dot(p1, inv(p2))
    R = rel_pose[:3, :3]
    t = rel_pose[:3, 3]

    return round(np.sqrt(np.linalg.norm(t) ** 2 + 2 * (1 - min(3.0, np.matrix.trace(R)) / 3)), 4)

class scannet(Dataset):
    
    def __init__(self, image_dir, label_dir, pose_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.pose_dir = pose_dir
        self.image_fns = os.listdir(image_dir)
        self.label_fns = os.listdir(label_dir)
        self.pose_fns = os.listdir(pose_dir)

    def __len__(self):
        return len(self.image_fns)
    
    def __getitem__(self, index):
        
        image_fn = self.image_fns[index]
        
        if index < (len(self.image_fns)-1):
            image_fn2 = self.image_fns[index+1]
        else:
#             print("enter")
            image_fn2 = self.image_fns[index]
        label_fn = self.label_fns[index]
        pose_fn = self.pose_fns[index]
        
        image_fp = os.path.join(self.image_dir, image_fn)
        image = Image.open(image_fp).convert('RGB')
        image = np.array(image)
        label_fp = os.path.join(self.label_dir, image_fn.split('.')[0]+'.png')
        label = Image.open(label_fp)
        label = np.array(label)
        cityscape, label = image, label
        cityscape = self.transform(cityscape)
        label_class = torch.Tensor(label).long()

        pose_fp1 =  os.path.join(self.pose_dir, image_fn.split('.')[0]+'.txt')
        # pose_fp2 =  os.path.join(self.pose_dir, image_fn2.split('.')[0]+'.txt')
        # D = self.pose_process(pose_fp1,pose_fp2)
        # print(pose_fp1)
        # print()
        D = self.pose_process_dum(pose_fp1)
#         D = self.test()
        return cityscape, label_class, D
    
    def test(self):

        gt_poses = []
        with open('/home/latai/Documents/Master_thesis_v2/data/test4/pose.txt') as f:
            for l in f.readlines():
                l = l.strip('\n')
                gt_poses.append(np.array(l.split(' ')).astype(np.float32).reshape(4, 4))
        poses = [gt_poses[0], gt_poses[1]]
        D = torch.from_numpy(np.expand_dims(self.genDistM(poses), 0))
        return D

    def pose_process_dum(self, file1):
        
        files = [file1]
        gt_poses = []
        
        for i in files:
            with open(i) as f:
                lines = f.readlines()
            r_temp = []
            for j in range(len(lines)):
                r_temp_2 = []
                for i in range(len(lines[j].split(' '))):
                    r_temp_2.append(float(lines[j].split(' ')[i]))
                r_temp.append(r_temp_2)
            gt_poses.append(r_temp)
            
#         poses = [gt_poses[0], gt_poses[1]]
        return gt_poses

    def pose_process(self, file1, file2):
        
        files = [file1, file2]
        gt_poses = []
        
        for i in files:
            with open(i) as f:
                lines = f.readlines()
            r_temp = []
            for j in range(len(lines)):
                r_temp_2 = []
                for i in range(len(lines[j].split(' '))):
                    r_temp_2.append(float(lines[j].split(' ')[i]))
                r_temp.append(r_temp_2)
            gt_poses.append(r_temp)
            
        poses = [gt_poses[0], gt_poses[1]]
        D = torch.from_numpy(np.expand_dims(self.genDistM(poses), 0))
        return D
    
    def pose_distance(self, p1, p2):
        rel_pose = np.dot(np.linalg.inv(p1), p2)
        R = rel_pose[:3, :3]
        t = rel_pose[:3, 3]

        return round(np.sqrt(np.linalg.norm(t) ** 2 + 2 * (1 - min(3.0, np.matrix.trace(R)) / 3)), 4)
    
    def genDistM(self, poses):
        n = len(poses)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D[i, j] = self.pose_distance(poses[i], poses[j])
        return D

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

dataset = scannet(train_dir_color, train_dir_label, train_dir_pose)
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
    
    def forward(self, X, D):
        contracting_11_out = self.contracting_11(X) # [-1, 64, 256, 256]
        contracting_12_out = self.contracting_12(contracting_11_out) # [-1, 64, 128, 128]
        contracting_21_out = self.contracting_21(contracting_12_out) # [-1, 128, 128, 128]
        contracting_22_out = self.contracting_22(contracting_21_out) # [-1, 128, 64, 64]
        contracting_31_out = self.contracting_31(contracting_22_out) # [-1, 256, 64, 64]
        contracting_32_out = self.contracting_32(contracting_31_out) # [-1, 256, 32, 32]
        contracting_41_out = self.contracting_41(contracting_32_out) # [-1, 512, 32, 32]
        contracting_42_out = self.contracting_42(contracting_41_out) # [-1, 512, 16, 16]
        middle_out = self.middle(contracting_42_out) # [-1, 1024, 16, 16]
        
        # with open('/scratch/mkolpe2s/MT/code/result_v3/middle_out.pkl', 'wb') as fp:
        #       pickle.dump(middle_out.detach().cpu().numpy(), fp)
        # with open('/scratch/mkolpe2s/MT/code/result_v3/D.pkl', 'wb') as fp:
        #       pickle.dump(D, fp)

        gamma2 = nn.Parameter(torch.randn(1), requires_grad=True).float()
        ell = nn.Parameter(torch.randn(1), requires_grad=True).float()
        sigma2 = nn.Parameter(torch.randn(1), requires_grad=True).float()

        Y = middle_out

        b=1
        l,c,h,w = Y.size()

        Y = Y.view(l,-1).cpu().float()
        D = D.float()
        K = torch.exp(gamma2) * (1 + math.sqrt(3) * D / torch.exp(ell)) * torch.exp(-math.sqrt(3) * D / torch.exp(ell))
        I = torch.eye(l).expand(l, l).float()
        X,_ = torch.solve(Y, K+torch.exp(sigma2)*I)
        Z = K.bmm(X)
        Z = F.relu(Z)
        new_list = []
        for i in range(len(D[0])):
            Z_new = Z[:, i].view(1, c, h, w)
            new_list.append(np.squeeze(Z_new.detach().numpy()))
        array = np.array(new_list)
        middle_out = torch.from_numpy(array).to(device)
        # gamma2 = nn.Parameter(torch.randn(1), requires_grad=True).float()
        # ell = nn.Parameter(torch.randn(1), requires_grad=True).float()
        # sigma2 = nn.Parameter(torch.randn(1), requires_grad=True).float()

        # b=1
        # l,c,h,w = middle_out.size()

        # Y = Y.view(l,-1).cpu().float()
        # D = D.float()
        # K = torch.exp(gamma2) * (1 + math.sqrt(3) * D / torch.exp(ell)) * torch.exp(-math.sqrt(3) * D / torch.exp(ell))
        # I = torch.eye(l).expand(l, l).float()
        # X,_ = torch.solve(Y, K+torch.exp(sigma2)*I)
        # Z = K.bmm(X)
        # Z = F.relu(Z)

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

num_classes = 41
batch_size = 4

epochs = 250
lr = 0.001

dataset = scannet(train_dir_color, train_dir_label, train_dir_pose)
data_loader = DataLoader(dataset, batch_size=batch_size)

with open('/scratch/mkolpe2s/MT/code/result_v2/data_loader.pkl', 'wb') as fp:
    pickle.dump(data_loader, fp)

model = UNet(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

step_losses = []
epoch_losses = []
for epoch in tqdm(range(epochs)):
    epoch_loss = 0
    for X, Y, D_dum in tqdm(data_loader, total=len(data_loader), leave=False):
        X, Y = X.to(device), Y.to(device)

        D1 = np.array([i.detach().numpy() for i in D_dum[0][0]])
        D2 = np.array([i.detach().numpy() for i in D_dum[0][1]])
        D3 = np.array([i.detach().numpy() for i in D_dum[0][2]])
        D4 = np.array([i.detach().numpy() for i in D_dum[0][3]])

        final_list  = []

        for i in range(len(X)):
            new_list = [D1[:,i],D2[:,i],D3[:,i],D4[:,i]]
            final_list.append(np.array(new_list))
        
        D = torch.from_numpy(np.expand_dims(genDistM(final_list), 0))

        optimizer.zero_grad()
        Y_pred = model(X, D)
        # Y = Y.squeeze(1)
        # Y = Y.cpu().detach().numpy()
        # Y_pred = Y_pred.cpu().detach().numpy()

        # with open('/scratch/mkolpe2s/MT/code/result_v2/Y.pkl', 'wb') as fp:
        #     pickle.dump(Y, fp)
        # with open('/scratch/mkolpe2s/MT/code/result_v2/Y_pred.pkl', 'wb') as fp:
        #     pickle.dump(Y_pred, fp)

        loss = criterion(Y_pred, Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        step_losses.append(loss.item())
        print("epoch_loss:",epoch_loss)
        print("step_losses:",loss.item() )
    text1 = "Epoch"+ entry.path.split('/')[-1].split('.')[0]
    bot.sendMessage(675791133, text)
    epoch_losses.append(epoch_loss/len(data_loader))

with open('/scratch/mkolpe2s/MT/code/result_v2/epoch_losses.pkl', 'wb') as fp:
    pickle.dump(epoch_losses, fp)

with open('/scratch/mkolpe2s/MT/code/result_v2/step_losses.pkl', 'wb') as fp:
    pickle.dump(step_losses, fp)

model_name = "/scratch/mkolpe2s/MT/code/result_v2/U-Net.pth"
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
        image_fp = os.path.join(self.image_dir, image_fn)
        image = Image.open(image_fp).convert('RGB')
        image = np.array(image)
        print(index)
        print(image_fp)
        label_fp = os.path.join(self.label_dir, image_fn.split('.')[0]+'.png')
        label = Image.open(label_fp)
        label = np.array(label)
        print(label_fp)
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

with open('/scratch/mkolpe2s/MT/code/result_v2/X.pkl', 'wb') as fp:
    pickle.dump(X, fp)

with open('/scratch/mkolpe2s/MT/code/result_v2/Y.pkl', 'wb') as fp:
    pickle.dump(Y, fp)

with open('/scratch/mkolpe2s/MT/code/result_v2/Y_pred.pkl', 'wb') as fp:
    pickle.dump(Y_pred, fp)