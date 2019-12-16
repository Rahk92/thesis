import torch
import torch.nn as nn
from torchvision.models import vgg16
from collections import namedtuple

class VGG16(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class PerceptualLoss(nn.Module):
    def __init__(self, device, weight=1):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGG16(requires_grad=False).to(device)
        self.loss = nn.MSELoss()
        self.weight = weight

    def forward(self, output, target):
        f1_output = self.vgg(output).relu1_2
        f1_target = self.vgg(target).relu1_2
        f1 = self.loss(f1_output, f1_target).item()

        f2_output = self.vgg(output).relu2_2
        f2_target = self.vgg(target).relu2_2
        f2 = self.loss(f2_output, f2_target).item()

        f3_output = self.vgg(output).relu3_3
        f3_target = self.vgg(target).relu3_3
        f3 = self.loss(f3_output, f3_target).item()

        sum = f1 + f2 + f3
        result = self.weight * sum
        return result