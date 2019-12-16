import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from models.p_loss import PerceptualLoss as PLoss
from torchsummary import summary
import models.models as models
import models.new_model as new_model
from PIL import Image
from pytorch_ssim import ssim as SSIM

from h5data import H5DataPrepare
from multi_crop import FolderDataset
from thop import profile

h5_path = 'E:/dataset/dehazing/dcpdn/val512'
test_path = 'E:/dataset/dehazing/mydataset/Valid/2018/'
result_path = './result/MDN3/2018'
load_weight_path = "./wgts/MDN3/train/859.ckpt"
epoch = os.path.basename(load_weight_path)

if not os.path.isdir(result_path):
    os.makedirs(result_path)

# Setting Train / Validation DataLoader
'''
val_dataset = H5DataPrepare(h5_path)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)
'''
val_dataset = FolderDataset(test_path + '/hazy/', test_path + '/GT/', valid=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1)

if torch.cuda.is_available:
    device = torch.device('cuda:0')
else :
    device = torch.device('cpu')

# Load model
model = new_model.MDN()
model = model.to(device)
summary(model, (3, 512, 512))
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(params)
input = torch.randn(1,3,512,512).to(device)
flops, params = profile(model, inputs=(input, ))
print("%.2f | %.2f" % (params / (1000 ** 2), flops / (1000 ** 3)))

try:
    model.load_state_dict(torch.load(load_weight_path))
    print("Load weight '{}'".format(load_weight_path))
except Exception as e:
    print("No weights. Exit the program...")
    sys.exit(1)

def psnr(output, target):
    mse_criterion = nn.MSELoss()
    mse = mse_criterion(output, target)
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr

vPSNR_sum = 0
vSSIM_sum = 0
voutput_list = []
model.train(False)
model.eval()
print("::: Testing start :::")
start_t = time.time()
with torch.no_grad():
    for i, (name, vhaze, vimage, (w1, w2, h1, h2)) in enumerate(val_loader):
        vhaze = vhaze.to(device)
        vimage = vimage.to(device)
        _, _, voutput = model(vhaze)
        voutput = F.pad(voutput, (-w1, -w2, -h1, -h2), mode='reflect')
        vimage = F.pad(vimage, (-w1, -w2, -h1, -h2), mode='reflect')
        voutput_list.append(voutput)
        print("{} PSNR : {:.2f}, SSIM : {:.4f}".format(name, psnr(voutput, vimage).item(),
                                                       SSIM(voutput, vimage).item()))
        vPSNR_sum += psnr(voutput, vimage).item()
        vSSIM_sum += SSIM(voutput, vimage).item()
    end_t = time.time()
    avg_vPSNR = vPSNR_sum / len(val_loader)
    avg_vSSIM = vSSIM_sum / len(val_loader)
    vmsg = "Average Testing PSNR : {:.6f}, SSIM : {:.6f}".format(avg_vPSNR, avg_vSSIM)
    print(vmsg)
    print("Running time : {:.2f}".format(end_t - start_t))
    print("Testing result saving.....")
    for j, dehazed in enumerate(voutput_list):
        dehazed = dehazed.cpu().detach()
        dehazed = np.array(np.clip(dehazed[0, :, :, :] * 255, 0, 255)).astype(np.uint8)
        dehazed = np.rollaxis(dehazed, 0, 3)
        vresult = transforms.ToPILImage()(dehazed)
        #vresult.show()
        vresult.save(result_path + "/" + epoch + "_" + str(j + 1) + ".png")
    print(":::Save Complete:::")
del voutput_list
torch.cuda.empty_cache()