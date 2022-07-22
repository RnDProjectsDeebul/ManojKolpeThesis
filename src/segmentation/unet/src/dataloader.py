import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

class data(Dataset):
    
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
        label_fp = os.path.join(self.label_dir, image_fn.split('.')[0]+'.png')
        label = Image.open(label_fp)
        label = np.array(label)
        cityscape, label = image, label
        cityscape = self.transform(cityscape)
        label_class = torch.Tensor(label).long()

        pose_fp1 =  os.path.join(self.pose_dir, image_fn.split('.')[0]+'.txt')
        D = self.pose_process_dum(pose_fp1)
        return cityscape, label_class, D, image_fn
    

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