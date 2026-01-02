import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

class PPM(nn.Module):
    def __init__(self, in_channels, reduce_dim):
        super(PPM, self).__init__()
        self.features = []
        self.scales = [1, 2, 3, 6]
        self.ppm_layers = nn.ModuleList()

        for scale in self.scales:
            self.ppm_layers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, reduce_dim, kernel_size=1, bias=False), 
                nn.BatchNorm2d(reduce_dim),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        x_size = x.size()[2:]
        output_list = [x]
        
        for layer in self.ppm_layers:
            out = layer(x)
            out = F.interpolate(out, size=x_size, mode='bilinear', align_corners=False)
            output_list.append(out)
            
        return torch.cat(output_list, dim=1)

class PSPNet(nn.Module):
    def __init__(self, num_classes=3, pretrained=True): 
        super(PSPNet, self).__init__()
        
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
        else:
            weights = None

        resnet = models.resnet50(
            weights=weights, 
            replace_stride_with_dilation=[False, True, True]
            )
        
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.ppm = PPM(in_channels=2048, reduce_dim=512)
        
        self.cls = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.size()[2:]
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.ppm(x)
        x = self.cls(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x