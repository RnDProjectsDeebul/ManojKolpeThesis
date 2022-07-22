"""
Imported packages
"""

from u_net_vanilla import UNet
from dataloader import data
from metric import IoU
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import wandb
from torch.utils.data import Dataset, DataLoader
wandb.login()

# ================================================================#
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print("Device:", device)
# ================================================================#
pretrained_model = "/content/drive/MyDrive/Master_thesis/Unet/unet_scannet_MT/result_unet_scannet_all_data_2_sequence/300_U-Net_gp.pth"
pretrained_epochs = 300
load_pretrain = True
path_to_save_model = "/content/drive/MyDrive/Master_thesis/Unet/unet_scannet_MT/result_unet_scannet_all_data_2_sequence/"
# ================================================================#
paths = []
train_sequence = []
validate_sequence = []

selected_seq_train =  False
path_to_all_data = "/content/drive/MyDrive/Master_thesis/Dataset/Scannet/scannet_processed_data_n"
path_to_train_data = "/content/drive/MyDrive/Master_thesis/Dataset/Scannet/scannet_processed_data_n"
path_to_validate_data = "/content/drive/MyDrive/Master_thesis/Dataset/Scannet/scannet_processed_data_n"

if not selected_seq_train:
    # Train with all train and validate sequences
    train_all_seq = True
    
    scenes = os.listdir(path_to_train_data)

    for i in scenes:
        text1 = path_to_train_data+i+"/output/color"
        text2 = path_to_train_data+i+"/output/label"
        text3 = path_to_train_data+i+"/output/pose"
        train_sequence.append((text1, text2, text3))
    
    scenes = os.listdir(path_to_validate_data)

    for i in scenes:
        text1 = path_to_validate_data+i+"/output/color"
        text2 = path_to_validate_data+i+"/output/label"
        text3 = path_to_validate_data+i+"/output/pose"
        validate_sequence.append((text1, text2, text3))

else:
    # Train and validate with selected sequences
    train_all_seq = False
    list_of_seq_train = ['scene0000_00', 'scene0000_01']
    list_of_seq_validate = ['scene0000_02']

    scenes = os.listdir(path_to_all_data)

    for i in scenes:
        text1 = path_to_all_data+i+"/output/color"
        text2 = path_to_all_data+i+"/output/label"
        text3 = path_to_all_data+i+"/output/pose"
        paths.append((text1, text2, text3))

    for j, i in enumerate(paths):
        if i[0].split('/')[-3] in list_of_seq_train:
            train_sequence.append(i)
        if i[0].split('/')[-3] in list_of_seq_validate:
            validate_sequence.append(i)

# ================================================================#
num_classes = 41
batch_size = 4
epochs = 201
lr = 0.001
# ================================================================#

def train(train_sequence, pretrained_epochs,path_to_save_model):
    if load_pretrain:
        model = UNet(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(pretrained_model))
    else:
        model = UNet(num_classes=num_classes).to(device)

    name_of_the_run = "unet_scannet_all_data_2_sequence_vanilla"
    architecture = "UNET_vanilla"
    dataset = "Scannet"

    wandb.init(
        # Set the project where this run will be logged
        project="Semantic_segmentation", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=name_of_the_run,
        # Track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "architecture": architecture,
        "dataset": dataset,
        "epochs": epochs,
        })

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    step_losses = []
    epoch_losses = []

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0

        for j, i in enumerate(train_sequence):
            dataset = data(i[0], i[1], i[2])
            data_loader = DataLoader(dataset, batch_size=batch_size)
            print(j, i)

            for X, Y, D, image_fn in tqdm(data_loader, total=len(data_loader), leave=False):
                X, Y = X.to(device), Y.to(device)
                optimizer.zero_grad()
                Y_pred = model(X)
                loss = criterion(Y_pred, Y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                step_losses.append(loss.item())
                print("epoch_loss:",epoch_loss)
                print("step_losses:",loss.item() )
                wandb.log({"epoch_loss": epoch_loss, "step_losses": loss.item()})
            epoch_losses.append(epoch_loss/len(data_loader))
            wandb.log({"epoch_losses": epoch_loss/len(data_loader)})
                
        if epoch%100 == 0:
            model_name = path_to_save_model+str(epoch+pretrained_epochs)+"_U-Net_vanilla.pth"
            torch.save(model.state_dict(), model_name)
    return model

def plot_data(dataset, title, path_to_save_fig):
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
    values = []
    values.append("Experiment:"+title)
    values.append("\n")
    values.append("Counted:"+str(counted))
    values.append("\n")
    factor=1.0/sum(counted.values())
    normalised_d = {k: v*factor for k, v in counted.items() }

    values.append("normalised_d:"+str(normalised_d))
    values.append("-------------------------------")

    plt.figure(figsize=(16,9))
    ax = sns.barplot(list(normalised_d.keys()), list(normalised_d.values()))
    ax.set_title('Class label distribution of '+title)
    ax.set_ylabel('Normalized number of pixels per class')
    ax.set_xlabel('Lables')
    plt.savefig(path_to_save_fig+title+'.pdf')

    return values

def validate(model, validate_sequence):

    Y_output = []
    Y_gt = []

    with torch.no_grad():
        for j, i in enumerate(validate_sequence):
            dataset = data(i[0], i[1], i[2])
            data_loader = DataLoader(dataset, batch_size=batch_size)
            for X, Y, D, image_fn in tqdm(data_loader, total=len(data_loader), leave=False):
                    X, Y = X.to(device), Y.to(device)

                    Y_pred = model(X)
                    Y_pred = torch.argmax(Y_pred, dim=1)
                    Y_output.append(Y_pred)
                    Y_gt.append(Y)

        Y_predicted = Y_output[0]
        for i in Y_output[1:]:
            Y_predicted = torch.cat([Y_predicted, i], dim=0)

        Y_ground_truth = Y_gt[0]
        for i in Y_gt[1:]:
            Y_ground_truth = torch.cat([Y_ground_truth, i], dim=0)

        x = IoU(num_classes=41)
        x.add(Y_predicted, Y_ground_truth)
        iou, meaniou = x.value()

        print("iou: ",iou)
        print("meaniou: ",meaniou)
        iou = "iou:"+str(iou)
        meaniou = "meaniou:"+str(meaniou)
        values = ["Experiment: 300_U-Net_vanilla_scene0000_02_validation_data", "\n", iou, "\n", meaniou]
        values.append("-------------------------------")

        Y_ground_truth = Y_ground_truth.cpu().detach().numpy()
        Y_predicted = Y_predicted.cpu().detach().numpy()

        path_to_save_fig = '/content/drive/MyDrive/Master_thesis/Unet/unet_scannet_MT/result_unet_scannet_all_data_2_sequence/'
        title_gt = "300_U-Net_vanilla_scene0000_02_Y_ground_truth_vanilla"
        title_pred = "300_U-Net_vanilla_scene0000_02_Y_predicted_vanilla"
        values1 = plot_data(Y_ground_truth, title_gt, path_to_save_fig)
        values2 = plot_data(Y_predicted, title_pred, path_to_save_fig)
        values = values+values1+values2

        path_to_store_result = "/content/drive/MyDrive/Master_thesis/Unet/unet_scannet_MT/result_unet_scannet_all_data_2_sequence/300_U-Net_vanilla_scene0000_02_iou_and_pixel_distribution.txt"
        with open(path_to_store_result, 'w') as output:
            for row in values:
                output.write(str(row) + '\n')


