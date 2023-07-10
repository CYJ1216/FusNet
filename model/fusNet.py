import torch
import torch.nn as nn
import torch.nn.functional as F
from model.res2net import res2net50_v1b_26w_4s

from model.swin import swin_tiny_patch4_window7_224
from model.cbam import cbam
import torchvision


class FAMBlock(nn.Module):
    def __init__(self, channels):
        super(FAMBlock, self).__init__()

        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

        self.relu3 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x3 = self.conv3(x)
        x3 = self.relu3(x3)
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        out = x3 + x1

        return out


class Extract(nn.Module):
    def __init__(self, channels):
        super(Extract, self).__init__()

        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3,stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

        self.relu3 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x3 = self.conv3(x)
        x3 = self.relu3(x3)
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        out = x3 + x1

        return out

class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, n_filters, use_transpose=True):
        super(DecoderBottleneckLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels , 1)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        if use_transpose:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels , in_channels, 3, stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")

        self.conv3 = nn.Conv2d(in_channels , n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x



class FusNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self):
        super(FusNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)  #特征提取层
        # ---- swin transformer Backbone ----
        self.swin = swin_tiny_patch4_window7_224(pretrained=True)  # 特征提取层

        self.x5_dem_1 = nn.Sequential(nn.Conv2d(2816, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.x4_dem_1 = nn.Sequential(nn.Conv2d(1408, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.x3_dem_1 = nn.Sequential(nn.Conv2d(704, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.x2_dem_1 = nn.Sequential(nn.Conv2d(352, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))  #第一层加强层
        self.x5_x4 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.x4_x3 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.x3_x2 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.x2_x1 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))

        self.x5_x4_x3 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.x4_x3_x2 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.x3_x2_x1 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))

        self.x5_x4_x3_x2 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96),nn.ReLU(inplace=True))
        self.x5_dem_4 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.x5_x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96),nn.ReLU(inplace=True))

        self.level3 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.level2 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.level1 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.x5_dem_5 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96),nn.ReLU(inplace=True))
        self.output4 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.output3 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.output2 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.output1 = nn.Sequential(nn.Conv2d(96, 1, kernel_size=3, padding=1))

        self.cbam = cbam(96)
        self.FAMBlock = FAMBlock(channels=96)

    def forward(self, x):
        input = x

        # '''
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x1r = self.resnet.maxpool(x)      # bs, 64, 56, 56
        # ---- low-level features ----
        x2r = self.resnet.layer1(x1r)      # bs, 256, 56, 56
        x3r = self.resnet.layer2(x2r)     # bs, 512, 28, 28
        x4r = self.resnet.layer3(x3r)     # bs, 1024, 14, 14
        x5r = self.resnet.layer4(x4r)     # bs, 2048, 7, 7

        bs= x.shape[0]
        x, H, W = self.swin.patch_embed(input)  #x 输入shape[2,3,244,244]
        x = self.swin.pos_drop(x)
        x1s = x.view(bs,96,56,56)  # bs, 96, 56, 56

        # ---- low-level features ----
        x, H, W = self.swin.layers[0](x, H, W)
        x2s = x.view(bs,192,28,28)   # bs, 192, 28, 28
        x, H, W = self.swin.layers[1](x, H, W)
        x3s = x.view(bs, 384, 14, 14)  # bs, 384, 14, 14
        x, H, W = self.swin.layers[2](x, H, W)
        x4s = x.view(bs, 768, 7, 7)  # bs, 768, 7, 7
        x, H, W = self.swin.layers[3](x, H, W)
        x5s = x.view(bs, 768, 7, 7)  # bs, 768, 7, 7

        #融合模块
        x1 = torch.cat([x2r, x1s], dim=1)
        x2 = torch.cat([x3r, x2s], dim=1)
        x3 = torch.cat([x4r, x3s], dim=1)
        x4 = torch.cat([x5r, x4s], dim=1)


        #进入加强层
        x5_dem_1 = self.x5_dem_1(x4)   #MS模块第一层
        x4_dem_1 = self.x4_dem_1(x3)   #MS模块第一层
        x3_dem_1 = self.x3_dem_1(x2)   #MS模块第一层
        x2_dem_1 = self.x2_dem_1(x1)   #MS模块第一层
        #第一层减法单元
        x5_4 = self.x5_x4(abs(F.upsample(x5_dem_1,size=x3.size()[2:], mode='bilinear')-x4_dem_1))
        x4_3 = self.x4_x3(abs(F.upsample(x4_dem_1,size=x2.size()[2:], mode='bilinear')-x3_dem_1))
        x3_2 = self.x3_x2(abs(F.upsample(x3_dem_1,size=x1.size()[2:], mode='bilinear')-x2_dem_1))
        x2_1 = self.x2_x1(F.upsample(x2_dem_1,size=x1.size()[2:], mode='bilinear'))

        #第二层减法单元
        x5_4_3 = self.x5_x4_x3(abs(F.upsample(x5_4, size=x4_3.size()[2:], mode='bilinear') - x4_3))
        x4_3_2 = self.x4_x3_x2(abs(F.upsample(x4_3, size=x3_2.size()[2:], mode='bilinear') - x3_2))
        x3_2_1 = self.x3_x2_x1(abs(F.upsample(x3_2, size=x2_1.size()[2:], mode='bilinear') - x2_1))

        #第三层减法单元
        x5_4_3_2 = self.x5_x4_x3_x2(abs(F.upsample(x5_4_3, size=x4_3_2.size()[2:], mode='bilinear') - x4_3_2))
        x4_3_2_1 = self.x4_x3_x2_x1(abs(F.upsample(x4_3_2, size=x3_2_1.size()[2:], mode='bilinear') - x3_2_1))
        #第四层减法单元
        x5_dem_4 = self.x5_dem_4(x5_4_3_2)
        x5_4_3_2_1 = self.x5_x4_x3_x2_x1(abs(F.upsample(x5_dem_4, size=x4_3_2_1.size()[2:], mode='bilinear') - x4_3_2_1))

        #特征融合单元
        level4 = x5_4
        level3 = self.level3(x4_3 + x5_4_3)   #不同尺度的融合1
        level2 = self.level2(x3_2 + x4_3_2 + x5_4_3_2) #不同尺度的融合2
        level1 = self.level1(x2_1 + x3_2_1 + x4_3_2_1 + x5_4_3_2_1)  #不同尺度的融合3

        #解码层 增加注意力模块，并进行堆叠特征
        # x5_4 = self.FAMBlock(x5_4)
        # level4 = self.FAMBlock(level4)
        # level3 = self.FAMBlock(level3)
        # level2 = self.FAMBlock(level2)
        # level1 = self.FAMBlock(level1)

        x5_dem_5 = self.x5_dem_5(x5_4)
        output4 = self.output4(F.upsample(x5_dem_5,size=level4.size()[2:], mode='bilinear') + level4)
        output3 = self.output3(F.upsample(output4,size=level3.size()[2:], mode='bilinear') + level3)
        output2 = self.output2(F.upsample(output3,size=level2.size()[2:], mode='bilinear') + level2)
        output1 = self.output1(F.upsample(output2,size=level1.size()[2:], mode='bilinear') + level1)


        # x5_dem_5 = self.x5_dem_5(self.cbam(x5_4))
        # output4 = self.output4(F.upsample(x5_dem_5,size=level4.size()[2:], mode='bilinear') + self.cbam(level4))
        # output3 = self.output3(F.upsample(output4,size=level3.size()[2:], mode='bilinear') + self.cbam(level3))
        # output2 = self.output2(F.upsample(output3,size=level2.size()[2:], mode='bilinear') + self.cbam(level2))
        # output1 = self.output1(F.upsample(output2,size=level1.size()[2:], mode='bilinear') + self.cbam(level1))

        output = F.upsample(output1, size=input.size()[2:], mode='bilinear')
        if self.training:
            return F.sigmoid(output)
        return F.sigmoid(output)  #shape [bs,1,224,224]




if __name__ == '__main__':
    ras = FusNet(n_classes=20).cuda()
    input_tensor = torch.randn(1, 3, 224, 224).cuda()  #原来是352
    out = ras(input_tensor)