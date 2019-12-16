import os
import glob
import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from models.p_loss import PerceptualLoss as PLoss
from torchsummary import summary
import models.models as models
import models.new_model as new_model
from PIL import Image

from h5data import H5DataPrepare
from multi_crop import FolderDataset

'''
# Load configuration
opt_file = open('./config.txt', "r")
opt = json.load(opt_file)
opt_file.close()

print('Running with following config:')
print(json.dumps(opt, indent=4, sort_keys=True))
'''
# Train configuration
batch_size = 2
aug = 10     #min == 1
num_epochs = 1000
learning_rate = 0.0001
early_stop = 0
early_stop_setting = 1000
restart = 0
fine = False
padd = 0
pad_setting = [0, 0]
val_save = False

train_path = 'E:/dataset/dehazing/Dense_Haze/Train'
val_path = 'E:/dataset/dehazing/Dense_Haze/Valid'
result_path = './result/MDN4'
log_path = "./log/MDN4"
weights_path = "./wgts/MDN4"
load_weight_path = "./wgts/MDN2/val/17.ckpt"

# Load weights
if not os.path.isdir(weights_path):
    os.makedirs(weights_path)
    os.makedirs(weights_path + '/train/')
    os.makedirs(weights_path + '/val/')
if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(result_path):
    os.makedirs(result_path)

# Setting Train / Validation DataLoader
'''
train_dataset = H5DataPrepare(train_path)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)
'''
# Data augmentation setting
transform_setting = transforms.Compose([
    transforms.RandomResizedCrop(512),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

train_dataset = FolderDataset(train_path + '/hazy/', train_path + '/GT/', transforms=transform_setting)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)
val_dataset = FolderDataset(val_path + '/hazy/', val_path + '/GT/', valid=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1)

if torch.cuda.is_available:
    device = torch.device('cuda:0')
else :
    device = torch.device('cpu')

# Load model
model = new_model.MDN()
model = model.to(device)
summary(model, (3, 512, 512))

if fine:
    print("Fine-tuning ON, continue from loaded weight")
    model.load_state_dict(torch.load(load_weight_path))
    print("Load weight '{}'".format(load_weight_path))
elif restart != 0:
    try:
        print("Restart ON, continue from epoch " + str(restart))
        model.load_state_dict(torch.load(weights_path + '/train/' + str(restart - 1) + '.ckpt'))
        print("Loading {}".format(str(restart) + '.ckpt'))
    except Exception as e:
        print("No weights. Training from scratch.")
        restart = 0
else:
    print("No weights. Training from scratch.")
    restart = 0

def psnr(output, target):
    mse_criterion = nn.MSELoss()
    mse = mse_criterion(output, target)
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr

# Criterion setting
criterion = {'L1':nn.L1Loss(),'MSE':nn.MSELoss(),'PLoss':PLoss(device, weight=0.3)}
train_loss = [criterion['MSE'], criterion['PLoss']]
val_loss = [criterion['MSE']]
best_loss = np.inf
alpha = 0.5
beta = 0.5
iteration = math.ceil(len(train_loader)) * aug

if padd:
    pad = nn.ReflectionPad2d((0, 0, pad_setting[0], pad_setting[1])).to(device)
    crop = nn.ReflectionPad2d((0, 0, -pad_setting[0], -pad_setting[1])).to(device)

writer = SummaryWriter(log_path)

# Training Loop
for epoch in range(restart, num_epochs):
    if epoch > 0 and epoch % 20 == 0:
        learning_rate = learning_rate / 2.0
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    iter = 1
    model.train()
    print("::: Training Step in Epoch {} :::".format(epoch + 1))
    tPSNR_sum = 0
    tloss_sum = 0
    for _ in range(aug):
        for _, (haze, image), _ in enumerate(train_loader):
            haze = haze.to(device)
            if padd:
                print("Padding and Cropping...")
                haze = pad(haze)
            image = image.to(device)
            a, t, output = model(haze)
            recon_haze = image * t + (1 - t) * a
            if padd:
                 output = crop(output)
            tloss = sum([ c(output, image) for c in train_loss ]) * alpha
            tloss += sum([ c(recon_haze, haze) for c in train_loss ]) * beta
            tloss_sum += tloss.item()
            tPSNR_sum += psnr(output, image).item()
            tmsg = "Epoch: [{}/{}], Iteration: [{}/{}], Loss: {:.6f},".format(
                epoch + 1, num_epochs, iter, iteration, tloss.item())
            print(tmsg)
            iter += 1
            optimizer.zero_grad()
            tloss.backward()
            optimizer.step()
    # Save log to tensorboard
    avg_tloss = tloss_sum / iteration
    avg_tPSNR = tPSNR_sum / iteration
    emsg = "< Epoch {} Result > Average Train Loss : {:.6f}, Average Train PSNR : {:.6f}".format(
        epoch + 1, avg_tloss, avg_tPSNR )
    print(emsg)
    writer.add_scalar('loss/train_loss', avg_tloss, epoch)
    writer.add_scalar('accuracy/train_PSNR', avg_tPSNR, epoch)
    torch.save(model.state_dict(), weights_path + "/train/" + str(epoch + 1) + ".ckpt")
    early_stop += 1

    # Validation stage
    print("::: Validation Step in Epoch {} :::".format(epoch + 1))
    vloss_sum = 0
    vPSNR_sum = 0
    voutput_list = []
    model.train(False)
    model.eval()
    with torch.no_grad():
        for _, (vhaze, vimage, (w1, w2, h1, h2)) in enumerate(val_loader):
            vhaze = vhaze.to(device)
            vimage = vimage.to(device)
            _, _, voutput = model(vhaze)
            voutput_list.append(voutput)
            vloss = sum([ c(voutput, vimage) for c in val_loss ])
            vloss_sum += vloss.item()
            vPSNR_sum += psnr(voutput, vimage).item()
        avg_vloss = vloss_sum / len(val_loader)
        avg_vPSNR = vPSNR_sum / len(val_loader)
        vmsg = "< Epoch {} Result > Average Validation Loss : {:.6f}, Average Validation PSNR : {:.6f}".format(
            epoch + 1, avg_vloss, avg_vPSNR)
        print(vmsg)
        writer.add_scalar('loss/val_loss', avg_vloss, epoch)
        writer.add_scalar('accuracy/val_PSNR', avg_vPSNR, epoch)

    if avg_vloss < best_loss:
        early_stop = 0
        best_loss = avg_vloss
        torch.save(model.state_dict(), weights_path + "/val/" + str(epoch + 1) + ".ckpt")
        print("Weight {}.ckpt saved".format(str(epoch + 1)))
        if val_save:
            print("Validation result saving.....")
            for j, dehazed in enumerate(voutput_list):
                dehazed = dehazed.cpu().detach()
                dehazed = np.array(np.clip(dehazed[0, :, :, :] * 255, 0, 255)).astype(np.uint8)
                dehazed = np.rollaxis(dehazed, 0, 3)
                vresult = transforms.ToPILImage()(dehazed)
                #vresult.show()
                vresult.save(result_path + "/" + str(epoch + 1) + "_" + str(j + 1) + ".png")
    del voutput_list
    torch.cuda.empty_cache()

    # Stop early if loss is not improving
    if early_stop == early_stop_setting:
        print("Loss has not improved in {} epochs. Stopping early.".format(early_stop_setting))
        break