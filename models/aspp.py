import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPModule(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding, dilation):
        super(ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.InstanceNorm2d(out_planes)
        self.relu = nn.PReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out

class ASPP(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = ASPPModule(in_planes, 16, 1, padding=0, dilation=dilations[0])
        self.aspp2 = ASPPModule(in_planes, 16, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = ASPPModule(in_planes, 16, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = ASPPModule(in_planes, 16, 3, padding=dilations[3], dilation=dilations[3])
        '''
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.insn = nn.InstanceNorm2d(in_planes)
        self.relu = nn.PReLU()
        self.conv = nn.Conv2d(in_planes, 20, kernel_size=1, stride=1, bias=False)
        '''
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_planes, 16, kernel_size=1, stride=1, bias=False),
                                             nn.InstanceNorm2d(16),
                                             nn.PReLU())
        self.conv = nn.Conv2d(16 * 5, out_planes, kernel_size=1, bias=False)
        self.bn = nn.InstanceNorm2d(out_planes)
        self.relu = nn.PReLU()
        self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        out1 = torch.cat((x1, x2, x3, x4, x5), dim=1)
        out2 = self.relu(self.bn(self.conv(out1)))
        #out = self.dropout(out2)
        return out2
