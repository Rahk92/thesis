import torch
import torch.nn as nn
from ps import PSBlock
from torch.nn import functional as F
from torchvision.models import resnet, vgg16, densenet121

class BottleNeckLayer(nn.Module):
    def __init__(self, in_planes, out_planes, droprate=0.0):
        super(BottleNeckLayer, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = droprate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, droprate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU()
        self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.droprate = droprate

    def forward(self, x):
        out = self.pool(self.conv(self.relu(self.bn(x))))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out


class Denseblock(nn.Module):
    def __init__(self, in_planes):
        super(Denseblock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_planes, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes + 32)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes + 32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_planes + 2*32)
        self.relu3 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_planes + 2*32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bottle = nn.Conv2d(in_planes + 3*32, in_planes, kernel_size=1, stride=1,
                                padding=0, bias=False)

    def forward(self, x):
        out1 = self.conv1(self.relu1(self.bn1(x)))
        out1 = torch.cat([x, out1], 1)
        out2 = self.conv2(self.relu2(self.bn2(out1)))
        out2 = torch.cat([out1, out2], 1)
        out3 = self.conv3(self.relu3(self.bn3(out2)))
        out3 = torch.cat([out2, out3], 1)
        out3 = self.bottle(out3)
        res = x + out3
        return res

class OriginDecoder(nn.Module):
    def __init__(self, out_channel):
        super(OriginDecoder, self).__init__()
        self.bottle = BottleNeckLayer(1024, 256)
        self.dense_block5 = Denseblock(256)
        self.dense_block6 = Denseblock(512 + 256)
        self.dense_block7 = Denseblock(256 + 128)
        self.dense_block8 = Denseblock(256)
        self.dense_block9 = Denseblock(128)

        self.upsample1 = PSBlock(512 + 256, 128, 2, kernel_size=3, stride=1, padding=1
                                 , activation='prelu', norm='batch')
        self.upsample2 = PSBlock(256 + 128, 128, 2, kernel_size=3, stride=1, padding=1
                                 , activation='prelu', norm='batch')
        self.upsample3 = PSBlock(256, 64, 2, kernel_size=3, stride=1, padding=1
                                 , activation='prelu', norm='batch')
        self.upsample4 = PSBlock(128, 32, 2, kernel_size=3, stride=1, padding=1
                                 , activation='prelu', norm='batch')
        self.upsample5 = PSBlock(16, 8, 2, kernel_size=3, stride=1, padding=1
                                 , activation='prelu', norm='batch')
        self.refine1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.refine2 = nn.Conv2d(8 + 3, 5, kernel_size=3, stride=1, padding=1)
        self.refine3 = nn.Conv2d(5, out_channel, kernel_size=3, stride=1, padding=1)
        self.ASPP = ASPP(5)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.relu3 = nn.PReLU()
        self.relu4 = nn.PReLU()
        self.tanh = nn.Tanh()

    def forward(self, I, x0, x1, x2, x3, x4, act=False):
        # [1,1024,16,16] to [1,256,16,16]
        out1 = self.bottle(x4)
        out1 = self.dense_block5(out1)

        # [1,256,16,16] to [1,128,32,32]
        out2 = self.dense_block6(torch.cat([x3, out1], 1))
        out2 = self.upsample1(out2)

        # [1,128,32,32] to [1,64,64,64]
        out3 = self.dense_block7(torch.cat([x2, out2], 1))
        out3 = self.upsample2(out3)

        # [1,64,64,64] to [1,64,128,128]
        out4 = self.dense_block8(torch.cat([x1, out3], 1))
        out4 = self.upsample3(out4)

        # [1,64,128,128] to [1,32,256,256]
        out5 = self.dense_block9(torch.cat([x0, out4], 1))
        out5 = self.upsample4(out5)

        # [1,32,256,256] to [1,8,512,512] with ASPP
        out6 = self.refine1(self.relu1(out5))
        out6 = self.relu2(self.upsample5(out6))

        out7 = self.relu3(self.refine2(torch.cat([out6, I], 1)))
        out7 = self.ASPP(out7)
        refine = self.refine3(out7)
        if act:
            result = self.tanh(refine)
        else:
            result = self.relu4(refine)

        return result

class Decoder(nn.Module):
    def __init__(self, out_channel):
        super(Decoder, self).__init__()
        self.bottle = BottleNeckLayer(1024, 256)
        self.dense_block5 = Denseblock(256)
        self.dense_block6 = Denseblock(512 + 256)
        self.dense_block7 = Denseblock(256 + 128)
        self.dense_block8 = Denseblock(256)
        self.dense_block9 = Denseblock(128)

        self.upsample1 = PSBlock(512 + 256, 128, 2, kernel_size=3, stride=1, padding=1
                                 ,activation='relu', norm='batch')
        self.upsample2 = PSBlock(256 + 128, 128, 2, kernel_size=3, stride=1, padding=1
                                 , activation='relu', norm='batch')
        self.upsample3 = PSBlock(256, 64, 2, kernel_size=3, stride=1, padding=1
                                 , activation='relu', norm='batch')
        self.upsample4 = PSBlock(128, 32, 2, kernel_size=3, stride=1, padding=1
                                 , activation='relu', norm='batch')
        self.upsample5 = PSBlock(32, 16, 2, kernel_size=3, stride=1, padding=1
                                 , activation='relu', norm='batch')
        self.refinement = nn.Conv2d(16, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x0, x1, x2, x3, x4):
        # [1,1024,16,16] to [1,256,16,16]
        out1 = self.bottle(x4)
        out1 = self.dense_block5(out1)

        # [1,256,16,16] to [1,128,32,32]
        out2 = self.dense_block6(torch.cat([x3, out1], 1))
        out2 = self.upsample1(out2)

        # [1,128,32,32] to [1,64,64,64]
        out3 = self.dense_block7(torch.cat([x2, out2], 1))
        out3 = self.upsample2(out3)

        # [1,64,64,64] to [1,64,128,128]
        out4 = self.dense_block8(torch.cat([x1, out3], 1))
        out4 = self.upsample3(out4)

        # [1,64,128,128] to [1,32,256,256]
        out5 = self.dense_block9(torch.cat([x0, out4], 1))
        out5 = self.upsample4(out5)

        # [1,32,256,256] to [1,out_channel,512,512]
        out6 = self.upsample5(out5)
        refine = self.refinement(out6)

        return refine


class MDN(nn.Module):
    def __init__(self):
        super(MDN, self).__init__()
        densenet = densenet121(pretrained=True)
        self.conv0 = densenet.features.conv0
        self.norm0 = densenet.features.norm0
        self.relu0 = densenet.features.relu0
        self.pool0 = densenet.features.pool0

        self.dense_block1 = densenet.features.denseblock1
        self.trans_block1 = densenet.features.transition1

        self.dense_block2 = densenet.features.denseblock2
        self.trans_block2 = densenet.features.transition2

        self.dense_block3 = densenet.features.denseblock3
        self.trans_block3 = densenet.features.transition3

        self.dense_block4 = densenet.features.denseblock4
        self.trans_block4 = TransitionBlock(1024, 1024)

        self.decoder_A = Decoder(out_channel=3)
        self.decoder_t = Decoder(out_channel=1)

    def forward(self, I):
        # [1,3,512,512] to [1,64,128,128]
        x0 = self.pool0(self.relu0(self.norm0(self.conv0(I))))
        # [1,64,128,128] to [1,128,64,64]
        x1 = self.trans_block1(self.dense_block1(x0))
        # [1,128,64,64 to [1,256,32,32]
        x2 = self.trans_block2(self.dense_block2(x1))
        # [1,256,32,32] to [1,512,16,16]
        x3 = self.trans_block3(self.dense_block3(x2))
        # [1,512,32,32] to [1,1024,16,16]
        x4 = self.dense_block4(x3)

        A = self.decoder_A(x0, x1, x2, x3, x4)
        t = self.decoder_t(x0, x1, x2, x3, x4)
        t = torch.abs((t)) + (10 ** -10)
        t = t.repeat(1, 3, 1, 1)

        recon_J = (I - (1-t) * A) / t

        return recon_J