#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Gz
# datetime： 2022/2/9 15:19 
import torch
from torchstat import stat
import torchvision.models as models
net = models.alexnet()
stat(net,(3,224,224))    # (3,224,224)表示输入图片的尺寸