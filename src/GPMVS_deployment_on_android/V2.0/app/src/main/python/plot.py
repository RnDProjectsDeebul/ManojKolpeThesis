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

def plot():
        # Encoder

    def down_conv_layer_en(input_channels, output_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=1,
                bias=False),
    nn.BatchNorm2d(output_channels),
    nn.ReLU(),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=2,
                bias=False),
    nn.BatchNorm2d(output_channels),
    nn.ReLU())

    def conv_layer(input_channels, output_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False),
    nn.BatchNorm2d(output_channels),
            nn.ReLU())

    def depth_layer(input_channels):
        return nn.Sequential(
            nn.Conv2d(input_channels, 1, 3, padding=1), nn.Sigmoid())

    def refine_layer(input_channels):
        return nn.Conv2d(input_channels, 1, 3, padding=1)

    def up_conv_layer(input_channels, output_channels, kernel_size):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False),
    nn.BatchNorm2d(output_channels),
            nn.ReLU())

    def get_trainable_number(variable):
        num = 1
        shape = list(variable.shape)
        for i in shape:
            num *= i
        return num

    class enCoder(nn.Module):

        def __init__(self):
            super(enCoder, self).__init__()

            self.conv1 = down_conv_layer_en(67, 128, 7)
            self.conv2 = down_conv_layer_en(128, 256, 5)
            self.conv3 = down_conv_layer_en(256, 512, 3)
            self.conv4 = down_conv_layer_en(512, 512, 3)
            self.conv5 = down_conv_layer_en(512, 512, 3)


        def getVolume(self, left_image, right_image, KRKiUV_T, KT_T):

            idepth_base = 1.0 / 50.0
            idepth_step = (1.0 / 0.5 - 1.0 / 50.0) / 63.0

            costvolume = Variable(
                torch.FloatTensor(left_image.shape[0], 64,
                                    left_image.shape[2], left_image.shape[3]))

            image_height = 256
            image_width = 320
            batch_number = left_image.shape[0]

            normalize_base = torch.FloatTensor(
                [image_width / 2.0, image_height / 2.0])

            normalize_base = normalize_base.unsqueeze(0).unsqueeze(-1)

            for depth_i in range(64):
                this_depth = 1.0 / (idepth_base + depth_i * idepth_step)
                transformed = KRKiUV_T * this_depth + KT_T
                demon = transformed[:, 2, :].unsqueeze(1)  
                warp_uv = transformed[:, 0: 2, :] / (demon + 1e-6)
                warp_uv = (warp_uv - normalize_base) / normalize_base
                warp_uv = warp_uv.view(
                    batch_number, 2, image_width,
                    image_height) 

                warp_uv = Variable(warp_uv.permute(
                    0, 3, 2, 1))  
                warped = F.grid_sample(right_image, warp_uv)

                costvolume[:, depth_i, :, :] = torch.sum(
                    torch.abs(warped - left_image), dim=1)
            return costvolume
            

        def forward(self, left_image, right_image, KRKiUV_T, KT_T):
            plane_sweep_volume = self.getVolume(left_image, right_image, KRKiUV_T, KT_T)

            x = torch.cat((left_image, plane_sweep_volume), 1)

            conv1 = self.conv1(x)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)

            conv5 = self.conv5(conv4)


            return [conv5, conv4, conv3, conv2, conv1]


    # Decoder

    def down_conv_layer(input_channels, output_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=1,
                bias=False),
    nn.BatchNorm2d(output_channels),
    nn.ReLU(),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=2,
                bias=False),
    nn.BatchNorm2d(output_channels),
    nn.ReLU())

    def conv_layer(input_channels, output_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False),
    nn.BatchNorm2d(output_channels),
            nn.ReLU())

    def depth_layer(input_channels):
        return nn.Sequential(
            nn.Conv2d(input_channels, 1, 3, padding=1), nn.Sigmoid())

    def refine_layer(input_channels):
        return nn.Conv2d(input_channels, 1, 3, padding=1)

    def up_conv_layer(input_channels, output_channels, kernel_size):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(

                input_channels,
                output_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False),
    nn.BatchNorm2d(output_channels),
            nn.ReLU())

    def get_trainable_number(variable):
        num = 1
        shape = list(variable.shape)
        for i in shape:
            num *= i
        return num

    class deCoder(nn.Module):


        def __init__(self):
            super(deCoder, self).__init__()

            self.upconv5 = up_conv_layer(512, 512, 3)
            self.iconv5 = conv_layer(1024, 512, 3)  #input upconv5 + conv4

            self.upconv4 = up_conv_layer(512, 512, 3)
            self.iconv4 = conv_layer(1024, 512, 3)  #input upconv4 + conv3
            self.disp4 = depth_layer(512)

            self.upconv3 = up_conv_layer(512, 256, 3)
            self.iconv3 = conv_layer(
                513, 256, 3)  #input upconv3 + conv2 + disp4 = 256 + 256 + 1 = 513
            self.disp3 = depth_layer(256)

            self.upconv2 = up_conv_layer(256, 128, 3)
            self.iconv2 = conv_layer(
                257, 128, 3)  #input upconv2 + conv1 + disp3 = 128 + 128 + 1 =  257
            self.disp2 = depth_layer(128)

            self.upconv1 = up_conv_layer(128, 64, 3)
            self.iconv1 = conv_layer(65, 64,
                                    3)  #input upconv1 + disp2 = 64 + 1 = 65
            self.disp1 = depth_layer(64)



        def forward(self, conv5, conv4, conv3, conv2, conv1):

            upconv5 = self.upconv5(conv5)

            iconv5 = self.iconv5(torch.cat((upconv5, conv4), 1))

            upconv4 = self.upconv4(iconv5)

            iconv4 = self.iconv4(torch.cat((upconv4, conv3), 1))
            disp4 = 2.0 * self.disp4(iconv4)
            udisp4 = F.upsample(disp4, scale_factor=2)


            upconv3 = self.upconv3(iconv4)

            iconv3 = self.iconv3(torch.cat((upconv3, conv2, udisp4), 1))
            disp3 = 2.0 * self.disp3(iconv3)
            udisp3 = F.upsample(disp3, scale_factor=2)


            upconv2 = self.upconv2(iconv3)

            iconv2 = self.iconv2(torch.cat((upconv2, conv1, udisp3), 1))
            disp2 = 2.0 * self.disp2(iconv2)
            udisp2 = F.upsample(disp2, scale_factor=2)


            upconv1 = self.upconv1(iconv2)
            iconv1 = self.iconv1(torch.cat((upconv1, udisp2), 1))
            disp1 = 2.0 * self.disp1(iconv1)

            if self.training:
                return [disp1, disp2, disp3, disp4]
            else:
                return disp1

    # Gaussian process

    class GPlayer(nn.Module):


        def __init__(self):
            super(GPlayer, self).__init__()

            self.gamma2 = nn.Parameter(torch.randn(1), requires_grad=True).float()
            self.ell = nn.Parameter(torch.randn(1), requires_grad=True).float()
            self.sigma2 = nn.Parameter(torch.randn(1), requires_grad=True).float()


        def forward(self, D, Y):
            '''
            :param D: Distance matrix
            :param Y: Stacked outputs from encoder
            :return: Z: transformed latent space
            '''
            b,l,c,h,w = Y.size()
            Y = Y.view(b,l,-1).cpu().float()
            D = D.float()
            print("Yup")
            K = torch.exp(self.gamma2) * (1 + math.sqrt(3) * D / torch.exp(self.ell)) * torch.exp(-math.sqrt(3) * D / torch.exp(self.ell))
            I = torch.eye(l).expand(b, l, l).float()

            X,_ = torch.solve(Y, K+torch.exp(self.sigma2)*I)

            Z = K.bmm(X)

            Z = F.relu(Z)

            return Z


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


    def compute_errors(gt, pred):
        valid1 = gt > 0.5
        valid2 = gt < 50
        valid = valid1 & valid2

        gt = gt[valid]
        pred = 1 / pred[valid]

        L1 = np.mean(np.abs(gt - pred))
        L1_rel = np.mean(np.abs(gt - pred) / gt)
        L1_inv = np.mean(np.abs(1 / gt - 1 / pred))

        log_diff = np.log(gt) - np.log(pred)
        sc_inv = np.sqrt(np.mean(np.square(log_diff)) - np.square(np.mean(log_diff)))

        return L1, L1_rel, L1_inv, sc_inv


    pixel_coordinate = np.indices([320, 256]).astype(np.float32)
    pixel_coordinate = np.concatenate(
        (pixel_coordinate, np.ones([1, 320, 256])), axis=0)
    pixel_coordinate = np.reshape(pixel_coordinate, [3, -1])

    def encoder_forward(r_img,n_img, r_pose,n_pose, K):

        left_image = r_img
        right_image = n_img


        left_pose = r_pose
        right_pose = n_pose

        camera_k = K


        left2right = np.dot(right_pose, inv(left_pose))

        # scale to 320x256
        original_width = left_image.shape[1]
        original_height = left_image.shape[0]
        factor_x = 320.0 / original_width
        factor_y = 256.0 / original_height

        left_image = cv2.resize(left_image, (320, 256))
        right_image = cv2.resize(right_image, (320, 256))
        camera_k[0, :] *= factor_x
        camera_k[1, :] *= factor_y

        # convert to torch
        torch_left_image = np.moveaxis(left_image, -1, 0)
        torch_left_image = np.expand_dims(torch_left_image, 0)

        torch_left_image = (torch_left_image - 81.0)/ 35.0
        torch_right_image = np.moveaxis(right_image, -1, 0)
        torch_right_image = np.expand_dims(torch_right_image, 0)

        torch_right_image = (torch_right_image - 81.0)/ 35.0


        left_image_cuda = Tensor(torch_left_image)
        left_image_cuda = Variable(left_image_cuda, volatile=True)

        right_image_cuda = Tensor(torch_right_image)
        right_image_cuda = Variable(right_image_cuda, volatile=True)

        left_in_right_T = left2right[0:3, 3]
        left_in_right_R = left2right[0:3, 0:3]
        K = camera_k
        K_inverse = inv(K)
        KRK_i = K.dot(left_in_right_R.dot(K_inverse))
        KRKiUV = KRK_i.dot(pixel_coordinate)
        KT = K.dot(left_in_right_T)
        KT = np.expand_dims(KT, -1)
        KT = np.expand_dims(KT, 0)
        KT = KT.astype(np.float32)
        KRKiUV = KRKiUV.astype(np.float32)
        KRKiUV = np.expand_dims(KRKiUV, 0)
        KRKiUV_cuda_T = Tensor(KRKiUV)
        KT_cuda_T = Tensor(KT)

        conv5, conv4, conv3, conv2, conv1= encoder(left_image_cuda, right_image_cuda, KRKiUV_cuda_T,KT_cuda_T)

        return conv5, conv4, conv3, conv2, conv1

    #load formatted sequence
    # scene = Path(seqpath)
    # intrinsics = np.loadtxt(scene / 'K.txt').astype(np.float32).reshape((3, 3))
    intrinsics = np.array([[585.,   0., 320.],
                          [  0., 585., 240.],
                          [  0.,   0.,   1.]])

    # imgs = sorted((scene/'images').files('*.png'))
    # gts = sorted((scene/'depth').files('*.npy'))

    # gt_poses = []
    # with open(scene / 'poses.txt') as f:
    #     for l in f.readlines():
    #         l = l.strip('\n')
    #         gt_poses.append(np.array(l.split(' ')).astype(np.float32).reshape(4, 4))

    # load pre-trained model

    pretrained_encoder = join(dirname(__file__), "encoder_model_best.pth.tar")
    pretrained_gplayer = join(dirname(__file__), "gp_model_best.pth.tar")
    pretrained_decoder = join(dirname(__file__), "decoder_model_best.pth.tar")

    encoder = enCoder()
    encoder = torch.nn.DataParallel(encoder)
    weights = torch.load(pretrained_encoder, map_location=torch.device('cpu'))
    encoder.load_state_dict(weights['state_dict'])
    encoder.eval()

    decoder = deCoder()
    decoder = torch.nn.DataParallel(decoder)
    weights = torch.load(pretrained_decoder, map_location=torch.device('cpu'))
    decoder.load_state_dict(weights['state_dict'])
    decoder.eval()

    gplayer =GPlayer()
    weights = torch.load(pretrained_gplayer, map_location=torch.device('cpu'))
    gplayer.load_state_dict(weights['state_dict'])
    gplayer.eval()

    # n = len(imgs)

    gt_poses = [np.array([[ 0.8825651 , -0.31022662,  0.35317943,  0.26405722],
       [ 0.3211478 ,  0.94653916,  0.02889591,  0.11636844],
       [-0.34328157,  0.08792515,  0.93504214, -0.46905556],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]), 
       np.array([[ 0.8744711 , -0.33191165,  0.3535954 ,  0.33358625],
       [ 0.33531535,  0.9405307 ,  0.05358316,  0.1364616 ],
       [-0.3503717 ,  0.07171355,  0.933794  , -0.53893393],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])]

    with torch.no_grad():
        poses = []
        idepths = []
        idepths_after = []
        latents = []
        conv1s = []
        conv2s = []
        conv3s = []
        conv4s = []

        preds = []

        r_pose = gt_poses[1]
        n_pose = gt_poses[1 - 1]

        filename1 = join(dirname(__file__), "0000.png")
        filename2 = join(dirname(__file__), "0001.png")

        r_img = cv2.imread(filename1)
        n_img = cv2.imread(filename2)

        # camera_k = np.loadtxt(scene / 'K.txt').astype(np.float32).reshape((3, 3))
        camera_k = intrinsics

        conv5, conv4, conv3, conv2, conv1 = encoder_forward(r_img, n_img, r_pose, n_pose, camera_k)

        poses.append(r_pose)

        latents.append(conv5)
        conv4s.append(conv4)
        conv3s.append(conv3)
        conv2s.append(conv2)
        conv1s.append(conv1)

        D = torch.from_numpy(np.expand_dims(genDistM(poses), 0))
        Y = torch.stack(latents, dim=1).cpu()

        Z = gplayer(D, Y)

        b, l, c, h, w = Y.size()

        
        conv5 = Z[:, 0].view(b, c, h, w)
        conv4 = conv4s[0]
        conv3 = conv3s[0]
        conv2 = conv2s[0]
        conv1 = conv1s[0]
        pred = decoder(conv5, conv4, conv3, conv2, conv1)

        idepths.append(pred[0][0].cpu().data.numpy())

        pred = cv2.resize(idepths[0], (640, 480))
        pred = np.clip(pred, a_min=0.02, a_max=2)  # depth range within [0.5, 50]
        preds.append(pred)

        # np.save(filename, np.array(preds))

        # pil_img = Image.fromarray(preds[0])
        # # pil_img = preds[0]

        # buff =  io.BytesIO()
        # pil_img.save(buff, format="PNG")
        # img_str = base64.b64decode(preds[0])

        plt.imshow(preds[0])

        f = io.BytesIO()
        plt.savefig(f, format="png")
    
    return f.getvalue()
