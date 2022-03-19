#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Gz
# datetime： 2021/11/3 15:58
import sys
import time

import torch.nn as nn
import torch
from torchsummary import summary
from torchvision import models
from functools import reduce

# from utils.cpu_info import get_platform_capability


class Vgg16(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True):
        super(Vgg16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),  # 1
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 2
            nn.ReLU(inplace=True),  # 3
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),  # 4
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 5
            nn.ReLU(inplace=True),  # 6
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 7
            nn.ReLU(inplace=True),  # 8
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),  # 9
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 10
            nn.ReLU(inplace=True),  # 11
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 12
            nn.ReLU(inplace=True),  # 13
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 14
            nn.ReLU(inplace=True),  # 15
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),  # 16
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 17
            nn.ReLU(inplace=True),  # 18
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 19
            nn.ReLU(inplace=True),  # 20
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 21
            nn.ReLU(inplace=True),  # 22
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),  # 23
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 24
            nn.ReLU(inplace=True),  # 25
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 26
            nn.ReLU(inplace=True),  # 27
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 28
            nn.ReLU(inplace=True),  # 29
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),  # 30
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # 18

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # 19
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),  # 20
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),  # 21
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x, partition=0, layerType="features"):
        # if collect == True:
        #     capability = get_platform_capability()
        # else:
        #     capability = None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if layerType == "features":
            layer = str(self.features[partition]).split('(')[0]
            input_data_size = sys.getsizeof(x.storage()) / 1024 / 1024
            x = self.features[partition](x)
            output_data_size = sys.getsizeof(x.storage()) / 1024 / 1024
        elif layerType == "classifier":
            input_data_size = sys.getsizeof(x.storage()) / 1024 / 1024
            layer = str(self.classifier[partition]).split('(')[0]
            x = self.classifier[partition](x)
            output_data_size = sys.getsizeof(x.storage()) / 1024 / 1024
        else:
            layer = "avgpool"
            input_data_size = sys.getsizeof(x.storage()) / 1024 / 1024
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            output_data_size = sys.getsizeof(x.storage()) / 1024 / 1024
        end.record()
        # 等待执行完成
        torch.cuda.synchronize()
        exec_time = start.elapsed_time(end)
        return x, layer, exec_time, input_data_size, output_data_size, partition

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


def vgg16(num_classes=1000, pretrained=True, progress=True):
    file = 'https://download.pytorch.org/models/vgg16-397923af.pth'
    model = Vgg16(num_classes)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(file, progress=progress)
        model.load_state_dict(state_dict)
    else:
        state_dict = torch.load('../../weights/vgg16-397923af.pth')
        model.load_state_dict(state_dict)
    return model


def record_latency(layer, latency, input_data_size, output_data_size, partition,layerType, imgSize):
    """保存参数"""
    res[layer]['Input Data Size(MB)'].append(input_data_size)
    res[layer]['Output Data Size(MB)'].append(output_data_size)
    if layerType =='features':
        res[layer]['Input Shape'].append(input_shapes[partition])
        res[layer]['Output Shape'].append(output_shapes[partition])
    else:
        res[layer]['Input Shape'].append(input_shapes[14+partition])
        res[layer]['Output Shape'].append(output_shapes[14+partition])
    res[layer]['Network'].append("vgg16")
    res[layer]['Platform'].append("edge server")
    res[layer]['ImgSize'].append(imgSize)
    res[layer]['Execute Time(ms)'].append(latency)


def execute_layers(intermediate, layerType, number, point, imgSize):
    """提取特征"""
    for partition in range(number):
        with torch.no_grad():
            if partition == point:
                _, _, _, _, _, _ = model(intermediate, partition=partition, layerType=layerType)
                intermediate, layer, t, input_data_size, output_data_size, p = model(intermediate,
                                                                                                 partition=partition,
                                                                                                 layerType=layerType)
            else:
                intermediate, layer, t, input_data_size, output_data_size, p = model(intermediate,
                                                                                                 partition=partition,
                                                                                                 layerType=layerType)
            print(layer, "===", t)
            print("============")
            times.append(t)
            layers.append(layer + "%s" % partition)
            output_data_sizes.append(output_data_size)
            input_data_sizes.append(input_data_size)
            record_latency(layer, t, input_data_size, output_data_size, partition,layerType, imgSize)

    return intermediate


def draw():
    import matplotlib.pyplot as plt
    total_width, n = 0.8, 2
    width = total_width / n
    with open("../../latencyRes/res_vgg16.txt", 'a', encoding='utf-8') as f:
        f.write(str(img_size) + "\n")
        f.write("layers: "+str(layers) + "\n")
        f.write("times: "+str(times) + "\n")
        f.write("input_data_sizes: "+str(input_data_sizes) + "\n")
        f.write("output_data_sizes: "+str(output_data_sizes) + "\n")
        f.write("input_shapes: "+str(input_shapes) + "\n")
        f.write("output_shapes: "+str(output_shapes) + "\n")
        f.write("\n")
    plt.bar([i for i in range(len(times))], times, width=width, label='Layer Latency')
    plt.bar([i + width for i in range(len(output_data_sizes))], output_data_sizes, width=width, tick_label=layers,
            label='Size of output data')
    plt.xticks(rotation=270)
    plt.title("vgg16 edge server %s * %s" % (img_size, img_size))
    plt.savefig("../../latencyRes/plt_edge/" + "vgg16_edge_server_%s.png" % img_size, dpi=600)
    plt.legend()
    plt.show()

def res_to_csv(header=True):
    import pandas as pd
    for key in list(res.keys()):
        df = pd.DataFrame(res[key])
        if header == False:
            df.to_csv('../../latencyRes/table_edge/%s_new.csv' % key, encoding='utf-8', header=False, index=False,mode='a')
        else:
            df.to_csv('../../latencyRes/table_edge/%s_new.csv' % key, encoding='utf-8',mode='a', index=False)


def process_summaries(summaries):
    for layer in summaries:
        input_shapes.append(abs(reduce(lambda x, y: x * y, summaries[layer]["input_shape"])))
        output_shapes.append(abs(reduce(lambda x, y: x * y, summaries[layer]["output_shape"])))

if __name__ == '__main__':
    print('test partition points in vgg16!!!')

    import json
    import torchvision.transforms as transforms
    from PIL import Image

    with open("../../data/imageNet_class_index.json", "r") as read_file:
        class_idx = json.load(read_file)
        labels = {int(key): value for key, value in class_idx.items()}

    model = vgg16(pretrained=True)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    imgSize = [224]
    for i in range(230, 641, 10):
        imgSize.append(i)
    for img_size in imgSize:
        print("输入图片尺寸为：", img_size)
        input_shapes = []
        output_shapes = []
        res = {key: {"Input Data Size(MB)": [], "Output Data Size(MB)": [], 'Input Shape': [],
                     'Output Shape': [], "Execute Time(ms)": [], "Network": [], "Platform": [], "ImgSize": []} for key in
               ['Conv2d', 'ReLU', 'MaxPool2d', 'Linear', 'Dropout']}
        summaries = summary(models.vgg16(False), input_size=(3, img_size, img_size), batch_size=-1,device='cpu')
        process_summaries(summaries)
        min_img_size = img_size
        transform_pipeline = transforms.Compose([transforms.Resize((min_img_size, min_img_size)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])

        img = Image.open('../../data/Golden_Retriever_Hund_Dog.jpg')
        img = transform_pipeline(img)
        img = img.unsqueeze(0)
        times = [0.01]
        layers = ["input"]
        output_data_sizes = [3 * min_img_size * min_img_size / 1024 / 1024]
        input_data_sizes = [3 * min_img_size * min_img_size / 1024 / 1024]

        intermediate = img.cuda()
        # 执行features
        intermediate = execute_layers(intermediate, "features", 31, 0,img_size)

        # avgpool
        intermediate, layer, t, input_data_size, output_data_size, p = model(intermediate, partition=0,
                                                                                         layerType="avgpool")
        print(layer, "===", t)
        print("============")
        times.append(t)
        layers.append(layer + "%s" % 11)
        output_data_sizes.append(output_data_size)

        # classifier
        intermediate = execute_layers(intermediate, "classifier", 7, 0,img_size)
        # print(res)
        # 绘图
        draw()
        if img_size == 224:
            res_to_csv(header=True)
        else:
            res_to_csv(header=False)
