import io
import matplotlib.pyplot as plt
import numpy as np
from path import Path
import cv2
from numpy.linalg import inv
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
from os.path import dirname, join
import os
from com.chaquo.python import Python
from PIL import Image
import base64
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.animation as animation

def plot():

    # Dataloader
    class scannet(Dataset):
    
        def __init__(self, image_dir):

            self.image_fns = image_dir

        def __len__(self):
            return len(self.image_fns)
        
        def __getitem__(self, index):
            
            image_fn = self.image_fns[index]
            
            image = Image.open(image_fn).convert('RGB')
            image = image.resize((320,240), Image.NEAREST)
            image = np.array(image)
            
            cityscape = image
            cityscape = self.transform(cityscape)

            return cityscape
        
        def transform(self, image):
            transform_ops = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            return transform_ops(image)

    # LSTM cell

    class MVSLayernormConvLSTMCell(nn.Module):

        def __init__(self, input_dim, hidden_dim, kernel_size, activation_function=None):
            super(MVSLayernormConvLSTMCell, self).__init__()

            self.activation_function = activation_function

            self.input_dim = input_dim
            self.hidden_dim = hidden_dim

            self.kernel_size = kernel_size
            self.padding = kernel_size[0] // 2, kernel_size[1] // 2

            self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                out_channels=4 * self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=False)

        def forward(self, input_tensor, cur_state):
            
            h_cur, c_cur = cur_state
            # print("h_next, c_next",h_cur.shape, c_cur.shape)
            non_valid = 0.01
            b, c, h, w = h_cur.size()

            non_valid = c

            combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
            # combined = input_tensor
            combined_conv = self.conv(combined)
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

            b, c, h, w = h_cur.size()
            
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)

            cc_g = torch.layer_norm(cc_g, [h, w])
            g = self.activation_function(cc_g)

            c_next = f * c_cur + i * g
            c_next = torch.layer_norm(c_next, [h, w])
            h_next = o * self.activation_function(c_next)

            return h_next, c_next

        def init_hidden(self, batch_size, image_size):
            height, width = image_size
            # 4, 512, 15, 20
            return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                    torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

    # Main function
    class UNet_lstm(nn.Module):
    
        def __init__(self, num_classes):
            super(UNet_lstm, self).__init__()
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

            self.current_state = None
            self.hyper_channels = 64

            input_size = self.hyper_channels * 16
            hidden_size = self.hyper_channels * 16

            self.lstm_cell = MVSLayernormConvLSTMCell(input_dim=input_size,
                                                    hidden_dim=hidden_size,
                                                    kernel_size=(3, 3),
                                                    activation_function=torch.celu)
            
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
            # print("X.shape", X.shape)
            batch, channel, height, width = middle_out.size()

            if self.current_state is None:
                # print("Entered here")
                hidden_state, cell_state = self.lstm_cell.init_hidden(batch_size=batch,
                                                                    image_size=(height, width))
            else:
                hidden_state, cell_state = self.current_state

            next_hidden_state, next_cell_state = self.lstm_cell(input_tensor=middle_out,
                                                                cur_state=[hidden_state, cell_state],
                                                                )
            hidden_state = next_hidden_state
            cell_state = next_cell_state
            # print("next_hidden_state.size:",next_hidden_state.size)
            
            # print(middle_out.shape)
            # RuntimeError: Given transposed=1, weight of size [1024, 512, 3, 3], expected input[4, 512, 15, 20] to have 1024 channels, but got 512 channels instead
            expansive_11_out = self.expansive_11(hidden_state) # [-1, 512, 32, 32]
            expansive_12_out = self.expansive_12(torch.cat((expansive_11_out, contracting_41_out), dim=1)) # [-1, 1024, 32, 32] -> [-1, 512, 32, 32]
            expansive_21_out = self.expansive_21(expansive_12_out) # [-1, 256, 64, 64]
            expansive_22_out = self.expansive_22(torch.cat((expansive_21_out, contracting_31_out), dim=1)) # [-1, 512, 64, 64] -> [-1, 256, 64, 64]
            expansive_31_out = self.expansive_31(expansive_22_out) # [-1, 128, 128, 128]
            expansive_32_out = self.expansive_32(torch.cat((expansive_31_out, contracting_21_out), dim=1)) # [-1, 256, 128, 128] -> [-1, 128, 128, 128]
            expansive_41_out = self.expansive_41(expansive_32_out) # [-1, 64, 256, 256]
            expansive_42_out = self.expansive_42(torch.cat((expansive_41_out, contracting_11_out), dim=1)) # [-1, 128, 256, 256] -> [-1, 64, 256, 256]
            output_out = self.output(expansive_42_out) # [-1, num_classes, 256, 256]
            return output_out

    # load pre-trained model

    color_valid = []

    path = join(dirname(__file__), "/data")

    for i in os.listdir(path):
        color_valid.append(path+i)

    dataset_valid = scannet(color_valid)
    data_loader_valid = DataLoader(dataset_valid, batch_size=8, shuffle=True)

    model_path = join(dirname(__file__), "2_U-Net_all_data_lstm.pth")

    model_lstm = UNet_lstm(num_classes=15)
    model_lstm.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    Y_output_lstm = []
    Y_gt_lstm = []

    for X in tqdm(data_loader_valid, total=len(data_loader_valid), leave=False):
        with torch.no_grad():

            Y_pred_lstm = torch.argmax(Y_pred_lstm, dim=1)
            Y_output_lstm.append(Y_pred_lstm.cpu().detach().numpy())
            Y_gt_lstm.append(Y.cpu().detach().numpy())

    fig, ax = plt.subplots()

    ims = []
    for i in range(30):
        x += np.pi / 15.
        y += np.pi / 20.
        im = ax.imshow(Y_output_lstm[i], animated=True)
        ims.append([im])

    animation_data = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)

    writer = animation.FFMpegWriter(fps=2, metadata=dict(artist='Me'), bitrate=1800)
    animation_data.save(join(dirname(__file__), "movie.mp4"), writer=writer)
    
    return None
