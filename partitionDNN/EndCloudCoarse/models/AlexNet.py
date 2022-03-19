#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Gz
# datetime： 2021/11/4 10:33
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, init_weights=True):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=3, out_channels=96, padding=(1, 1)),
            nn.BatchNorm2d(96),
            nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(kernel_size=5, in_channels=96, out_channels=256, padding=(2, 2)),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(kernel_size=3, in_channels=256, out_channels=384, padding=(1, 1)),
            nn.Conv2d(kernel_size=3, in_channels=384, out_channels=384, padding=(1, 1)),
            nn.Conv2d(kernel_size=3, in_channels=384, out_channels=256, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Linear(4 * 4 * 256, 2048),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.Dropout(),
            nn.Linear(1024, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x, partition=0):
        x = self.features[partition](x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def alexnet(num_classes=10, pretrained=True, progress=True):
    file = 'https://download.pytorch.org/models/vgg16-397923af.pth'
    model = AlexNet(num_classes)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(file, progress=progress)
        model.load_state_dict(state_dict)
    else:
        state_dict = torch.load('../weights/AlexNet_cifar10.pt')
        model.load_state_dict(state_dict)
    return model