import torch
import torch.nn as nn
from ps import PSBlock
from torch.nn import functional as F
from torchvision.models import resnet, vgg16, densenet121
from models.aspp import ASPP


class Denseblock(nn.Module):
    def __init__(self, in_planes):
        super(Denseblock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_planes + 32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_planes + 2 * 32, in_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.relu3 = nn.PReLU()

    def forward(self, x):
        out1 = self.relu1(self.bn1(self.conv1(x)))
        out1 = torch.cat([x, out1], 1)
        out2 = self.relu2(self.bn2(self.conv2(out1)))
        out2 = torch.cat([out1, out2], 1)
        out3 = self.relu3(self.bn3(self.conv3(out2)))
        res = torch.add(x, out3)
        return res

class TransBlock(nn.Module):
    def __init__(self, in_planes, out_planes, scale, dropRate=0.0):
        super(TransBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                                        padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.PReLU()
        self.scale = scale
        self.droprate = dropRate

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.interpolate(out, scale_factor=self.scale, mode='bilinear', align_corners=False)

class New1Decoder(nn.Module):
    def __init__(self, out_channel):
        super(New1Decoder, self).__init__()
        self.dense_block5 = Denseblock(512)
        self.dense_block6 = Denseblock(512)
        self.dense_block7 = Denseblock(256)
        self.dense_block8 = Denseblock(128)
        # upsampling using TConv
        self.upsample11 = TransBlock(512, 256, 2)
        self.upsample22 = TransBlock(512, 128, 2)
        self.upsample33 = TransBlock(256, 64, 2)
        self.upsample441 = TransBlock(128, out_channel, 4)
        self.upsample442 = TransBlock(128, out_channel, 4)
        # upsampling using Subpixel layer
        self.upsample1 = PSBlock(512, 256, 2, kernel_size=3, stride=1, padding=1
                                 , activation='prelu', norm='batch')
        self.upsample2 = PSBlock(512, 128, 2, kernel_size=3, stride=1, padding=1
                                 , activation='prelu', norm='batch')
        self.upsample3 = PSBlock(256, 64, 2, kernel_size=3, stride=1, padding=1
                                 , activation='prelu', norm='batch')
        self.upsample41 = PSBlock(128, out_channel, 4, kernel_size=3, stride=1, padding=1
                                 , activation='prelu', norm='batch')
        self.upsample42 = PSBlock(128, out_channel, 4, kernel_size=3, stride=1, padding=1
                                 , activation='prelu', norm='batch')
        self.ASPP = ASPP(64, 128)
        self.refine1 = nn.Conv2d(out_channel * 2, 16, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv20 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv30 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv40 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.refine2 = nn.Conv2d(16 + 4, out_channel, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.PReLU()
        self.relu21 = nn.PReLU()
        self.relu22 = nn.PReLU()
        self.relu23 = nn.PReLU()
        self.relu24 = nn.PReLU()
        self.relu3 = nn.PReLU()
        self.tanh = nn.Tanh()

    def forward(self, I, x0, x1, x2, x3):
        # [512,16,16] to [256,32,32]
        out1 = self.dense_block5(x3)
        out1 = self.upsample11(out1)

        # [256,32,32] to [128,64,64]
        out2 = self.dense_block6(torch.cat([x2, out1], 1))
        out2 = self.upsample22(out2)

        # [128,64,64] to [64,128,128]
        out3 = self.dense_block7(torch.cat([x1, out2], 1))
        out3 = self.upsample33(out3)

        # [64,128,128] to Atrous Spatial Pyramid Pooling
        out4 = self.dense_block8(torch.cat([x0, out3], 1))
        aspp = self.ASPP(x0)

        #Upsampling to output_size [out_channel,512,512]
        out41 = self.upsample441(out4)
        out42 = self.upsample442(aspp)
        out5 = torch.cat([out41, out42], 1)

        # Refinement
        out6 = self.relu1(self.refine1(out5))
        shape_out = out6.data.size()
        shape_out = shape_out[2:4]
        out61 = F.avg_pool2d(out6, 32)
        out62 = F.avg_pool2d(out6, 16)
        out63 = F.avg_pool2d(out6, 8)
        out64 = F.avg_pool2d(out6, 4)
        out610 = F.interpolate(self.relu21(self.conv10(out61)), size=shape_out, mode='bilinear', align_corners=False)
        out620 = F.interpolate(self.relu22(self.conv20(out62)), size=shape_out, mode='bilinear', align_corners=False)
        out630 = F.interpolate(self.relu23(self.conv30(out63)), size=shape_out, mode='bilinear', align_corners=False)
        out640 = F.interpolate(self.relu24(self.conv40(out64)), size=shape_out, mode='bilinear', align_corners=False)
        result = torch.cat((out610, out620, out630, out640, out6), 1)
        result = self.relu3(self.refine2(result))

        return result

class MDN(nn.Module):
    def __init__(self):
        super(MDN, self).__init__()
        densenet = densenet121(pretrained=True)
        #self.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=3, dilation=3)
        self.conv0 = densenet.features.conv0
        self.norm0 = densenet.features.norm0
        self.relu0 = nn.PReLU()
        self.pool0 = densenet.features.pool0

        self.dense_block1 = densenet.features.denseblock1
        self.trans_block1 = densenet.features.transition1

        self.dense_block2 = densenet.features.denseblock2
        self.trans_block2 = densenet.features.transition2

        self.dense_block3 = densenet.features.denseblock3
        self.trans_block3 = densenet.features.transition3

        self.dense_block4 = densenet.features.denseblock4

        self.decoder_A = New1Decoder(out_channel=3)
        self.decoder_t = New1Decoder(out_channel=1)
        self.sig = nn.Sigmoid()

    def forward(self, I):
        # [1,3,512,512] to [1,64,128,128]
        x0 = self.pool0(self.relu0(self.norm0(self.conv0(I))))
        # [1,64,128,128] to [1,128,64,64]
        x1 = self.trans_block1(self.dense_block1(x0))
        # [1,128,64,64 to [1,256,32,32]
        x2 = self.trans_block2(self.dense_block2(x1))
        # [1,256,32,32] to [1,512,16,16]
        x3 = self.trans_block3(self.dense_block3(x2))
        # [1,512,16,16] to [1,1024,16,16]
        #x4 = self.dense_block4(x3)

        A = self.decoder_A(I, x0, x1, x2, x3)
        t = self.decoder_t(I, x0, x1, x2, x3)
        t = self.sig(t)
        t = t.repeat(1, 3, 1, 1)

        J = torch.add(1.0, -t)
        J = torch.mul(J, A)
        J = torch.add(I, -J)
        recon_J = torch.div(J, t)

        return A, t, recon_J