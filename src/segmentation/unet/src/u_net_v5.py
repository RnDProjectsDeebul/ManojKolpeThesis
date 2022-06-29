import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from collections import Counter
import seaborn as sns
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
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from tqdm import tqdm
import telepot
bot = telepot.Bot('5297815301:AAF6dr1AfH4BGPu1DeJ0vE2vhddA3bMTkEg')

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print("Device:", device)

train_dir_color = '/scratch/mkolpe2s/MT/Main_data_folder/Segmentation_dataset/scannet/experiment_data/data2/rgb_train/'
train_dir_label = '/scratch/mkolpe2s/MT/Main_data_folder/Segmentation_dataset/scannet/experiment_data/data2/label_train/'
val_dir_color = '/scratch/mkolpe2s/MT/Main_data_folder/Segmentation_dataset/scannet/experiment_data/data2/rgb_test/'
val_dir_label = '/scratch/mkolpe2s/MT/Main_data_folder/Segmentation_dataset/scannet/experiment_data/data2/label_test/'
train_dir_pose = '/scratch/mkolpe2s/MT/Main_data_folder/Segmentation_dataset/scannet/experiment_data/data2/train_pose/'
val_dir_pose = '/scratch/mkolpe2s/MT/Main_data_folder/Segmentation_dataset/scannet/experiment_data/data2/test_pose/'

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
        label_fn = self.label_fns[index]
        pose_fn = self.pose_fns[index]
        
        image_fp = os.path.join(self.image_dir, image_fn)
        image = Image.open(image_fp).convert('RGB')
        image = np.array(image)
        image = image[:,:1239]
        label_fp = os.path.join(self.label_dir, "classgt_"+image_fn.split('.')[0].split('_')[1]+'.png')
        label = Image.open(label_fp)
        label = np.array(label)
        label = label[:,:1239]
        cityscape, label = image, label
        cityscape = self.transform(cityscape)
        label_class = torch.Tensor(label).long()

        pose_fp1 =  os.path.join(self.pose_dir, image_fn.split('.')[0].split('_')[1]+'.txt')
        D = self.pose_process_dum(pose_fp1)
        return cityscape, label_class, D
    

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
        return gt_poses
    
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
        self.expansive_21 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.expansive_22 = self.conv_block(in_channels=512, out_channels=256)
        self.expansive_31 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=256, out_channels=128)
        self.expansive_41 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, output_padding=1)
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

        Y = middle_out

        b=1
        l,c,h,w = Y.size()

        Y = Y.view(l,-1).cpu().float()
        D = D.float()
        K = torch.exp(self.gamma2) * (1 + math.sqrt(3) * D / torch.exp(self.ell)) * torch.exp(-math.sqrt(3) * D / torch.exp(self.ell))
        I = torch.eye(l).expand(l, l).float()
        
        X,_ = torch.solve(Y, K+torch.exp(self.sigma2)*I)
        Z = K.bmm(X)
        Z = F.relu(Z)
        new_list = []
        for i in range(len(D[0])):
            Z_new = Z[:, i].view(1, c, h, w)
            new_list.append(np.squeeze(Z_new.detach().numpy()))
        array = np.array(new_list)
        middle_out = torch.from_numpy(array).to(device)

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

num_classes = 15
batch_size = 4

epochs = 80
lr = 0.001

dataset = scannet(train_dir_color, train_dir_label, train_dir_pose)
data_loader = DataLoader(dataset, batch_size=batch_size)

with open('/scratch/mkolpe2s/MT/code/result_v5/data_loader.pkl', 'wb') as fp:
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

        loss = criterion(Y_pred, Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        step_losses.append(loss.item())
        print("epoch_loss:",epoch_loss)
        print("step_losses:",loss.item() )

    text1 = "U_net_v5 Epoch_80:"+ str(epoch)
    text2 = "Epoch_loss v5_80: "+str(epoch_loss)+" Step_losses v3: "+str(loss.item())
    text3 = "-------------------"
    bot.sendMessage(675791133, text1)
    bot.sendMessage(675791133, text2)
    bot.sendMessage(675791133, text3)
    epoch_losses.append(epoch_loss/len(data_loader))

with open('/scratch/mkolpe2s/MT/code/result_v5/epoch_losses.pkl', 'wb') as fp:
    pickle.dump(epoch_losses, fp)

with open('/scratch/mkolpe2s/MT/code/result_v5/step_losses.pkl', 'wb') as fp:
    pickle.dump(step_losses, fp)

model_name = "/scratch/mkolpe2s/MT/code/result_v5/U-Net.pth"
torch.save(model.state_dict(), model_name)

torch.cuda.empty_cache()

test_batch_size = 4
dataset = scannet(val_dir_color, val_dir_label, val_dir_pose)
data_loader = DataLoader(dataset, batch_size=test_batch_size)

X, Y, D = next(iter(data_loader))
X, Y = X.to(device), Y.to(device)

D1 = np.array([i.detach().numpy() for i in D[0][0]])
D2 = np.array([i.detach().numpy() for i in D[0][1]])
D3 = np.array([i.detach().numpy() for i in D[0][2]])
D4 = np.array([i.detach().numpy() for i in D[0][3]])
final_list  = []

for i in range(len(X)):
    new_list = [D1[:,i],D2[:,i],D3[:,i],D4[:,i]]
    final_list.append(np.array(new_list))

D_new = torch.from_numpy(np.expand_dims(genDistM(final_list), 0))

Y_pred = model(X, D_new)
Y_pred = torch.argmax(Y_pred, dim=1)

inverse_transform = transforms.Compose([
    transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
])

X = X.cpu().detach().numpy()
Y = Y.cpu().detach().numpy()
Y_pred = Y_pred.cpu().detach().numpy()

print(X.shape, Y.shape, Y_pred.shape)

with open('/scratch/mkolpe2s/MT/code/result_v5/X.pkl', 'wb') as fp:
    pickle.dump(X, fp)

with open('/scratch/mkolpe2s/MT/code/result_v5/Y.pkl', 'wb') as fp:
    pickle.dump(Y, fp)

with open('/scratch/mkolpe2s/MT/code/result_v5/D.pkl', 'wb') as fp:
    pickle.dump(Y, fp)

with open('/scratch/mkolpe2s/MT/code/result_v5/Y_pred.pkl', 'wb') as fp:
    pickle.dump(Y_pred, fp)

inverse_transform = transforms.Compose([
    transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
])

fig, axes = plt.subplots(test_batch_size, 3, figsize=(3*5, test_batch_size*5))

for i in range(test_batch_size):
    
    landscape = inverse_transform(torch.from_numpy(X[i])).permute(1, 2, 0).cpu().detach().numpy()
    label_class = Y[i]
    label_class_predicted = Y_pred[i]
    
    axes[i, 0].imshow(landscape)
    axes[i, 0].set_title("Landscape")
    axes[i, 1].imshow(label_class)
    axes[i, 1].set_title("Label Class")
    axes[i, 2].imshow(label_class_predicted)
    axes[i, 2].set_title("Label Class - Predicted")

plt.savefig('/scratch/mkolpe2s/MT/code/result_v5/validation.pdf')

class Metric(object):
    """Base class for all metrics.

    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py
    """
    def reset(self):
        pass

    def add(self):
        pass

    def value(self):
        pass
class ConfusionMatrix(Metric):
    """Constructs a confusion matrix for a multi-class classification problems.

    Does not support multi-label, multi-class problems.

    Keyword arguments:
    - num_classes (int): number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.

    Modified from: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
    """

    def __init__(self, num_classes, normalized=False):
        super().__init__()

        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int64)
        self.normalized = normalized
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix

        The shape of the confusion matrix is K x K, where K is the number
        of classes.

        Keyword arguments:
        - predicted (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        predicted scores obtained from the model for N examples and K classes,
        or an N-tensor/array of integer values between 0 and K-1.
        - target (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        ground-truth classes for N examples and K classes, or an N-tensor/array
        of integer values between 0 and K-1.

        """
        # If target and/or predicted are tensors, convert them to numpy arrays
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.num_classes, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.num_classes) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        if np.ndim(target) != 1:
            assert target.shape[1] == self.num_classes, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (target.max() < self.num_classes) and (target.min() >= 0), \
                'target values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(
            x.astype(np.int64), minlength=self.num_classes**2)
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf

class IoU(Metric):
    """Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU).

    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:

        IoU = true_positive / (true_positive + false_positive + false_negative).

    Keyword arguments:
    - num_classes (int): number of classes in the classification problem
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    - ignore_index (int or iterable, optional): Index of the classes to ignore
    when computing the IoU. Can be an int, or any iterable of ints.
    """

    def __init__(self, num_classes, normalized=False, ignore_index=None):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized)

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        """Adds the predicted and target pair to the IoU metric.

        Keyword arguments:
        - predicted (Tensor): Can be a (N, K, H, W) tensor of
        predicted scores obtained from the model for N examples and K classes,
        or (N, H, W) tensor of integer values between 0 and K-1.
        - target (Tensor): Can be a (N, K, H, W) tensor of
        target scores for N examples and K classes, or (N, H, W) tensor of
        integer values between 0 and K-1.

        """
        # Dimensions check
        assert predicted.size(0) == target.size(0), \
            'number of targets and predicted outputs do not match'
        assert predicted.dim() == 3 or predicted.dim() == 4, \
            "predictions must be of dimension (N, H, W) or (N, K, H, W)"
        assert target.dim() == 3 or target.dim() == 4, \
            "targets must be of dimension (N, H, W) or (N, K, H, W)"

        # If the tensor is in categorical format convert it to integer format
        if predicted.dim() == 4:
            _, predicted = predicted.max(1)
        if target.dim() == 4:
            _, target = target.max(1)

        self.conf_metric.add(predicted.view(-1), target.view(-1))

    def value(self):
        """Computes the IoU and mean IoU.

        The mean computation ignores NaN elements of the IoU array.

        Returns:
            Tuple: (IoU, mIoU). The first output is the per class IoU,
            for K classes it's numpy.ndarray with K elements. The second output,
            is the mean IoU.
        """
        conf_matrix = self.conf_metric.value()
        if self.ignore_index is not None:
            conf_matrix[:, self.ignore_index] = 0
            conf_matrix[self.ignore_index, :] = 0
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)

        return iou, np.nanmean(iou)

x = IoU(num_classes=41)
x.add(torch.from_numpy(Y_pred), torch.from_numpy(Y))
iou, meaniou = x.value()
print("iou: ",iou)
print("meaniou: ",meaniou)

def plot_data(dataset, title):
#     dataset = "/home/latai/Documents/Master_thesis_v2/data/test4/label_train_n/"

    classes = []
    for i in dataset:
        classes.append(i.flatten())

    print("Number of images:", len(classes))
    print("Shape of single image:", dataset[0].shape)

    flat_list = [x for xs in classes for x in xs]
    print("Total number of pixel in entire dataset:", len(flat_list))
    counted = dict(Counter(flat_list))
    print("Label data:")
    print(counted)

    factor=1.0/sum(counted.values())
    normalised_d = {k: v*factor for k, v in counted.items() }

    plt.figure(figsize=(16,9))
    ax = sns.barplot(list(normalised_d.keys()), list(normalised_d.values()))
    ax.set_title('Class label distribution of '+title)
    ax.set_ylabel('Normalized number of pixels per class')
    ax.set_xlabel('Lables')
    plt.savefig('/scratch/mkolpe2s/MT/code/result_v5/'+title+'.pdf')

plot_data(Y, "Y")
plot_data(Y_pred, "Y_pred")
