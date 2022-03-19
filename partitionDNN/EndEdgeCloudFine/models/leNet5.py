#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Gz
# datetime： 2021/11/4 10:33
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models
from functools import reduce
from EndEdgeCloudFine.models.lenet_summ import lenets


# from utils.cpu_info import get_platform_capability


class LeNet(nn.Module):
    def __init__(self, num_classes=10, init_weights=True):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(16 * 6 * 6, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x, partition=0, layerType="features", collect=False):
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


def lenet(num_classes=10, pretrained=True, progress=True):
    file = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
    model = LeNet(num_classes)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(file, progress=progress)
        model.load_state_dict(state_dict)
    else:
        state_dict = torch.load('../../weights/cifar_net.pth')
        model.load_state_dict(state_dict)
    return model


def record_latency(layer, latency, input_data_size, output_data_size, partition, layerType, imgSize):
    """保存参数"""
    res[layer]['Input Data Size(MB)'].append(input_data_size)
    res[layer]['Output Data Size(MB)'].append(output_data_size)
    # if layerType == 'features':
    #     res[layer]['Input Shape'].append(input_shapes[partition])
    #     res[layer]['Output Shape'].append(output_shapes[partition])
    # else:
    #     res[layer]['Input Shape'].append(input_shapes[14 + partition])
    #     res[layer]['Output Shape'].append(output_shapes[14 + partition])
    res[layer]['Network'].append("Alexnet")
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
                                                                                                 layerType=layerType,
                                                                                                 collect=False)
            else:
                intermediate, layer, t, input_data_size, output_data_size, p = model(intermediate,
                                                                                                 partition=partition,
                                                                                                 layerType=layerType,
                                                                                                 collect=False)
            print(layer, "===", t)
            print("============")
            times.append(t)
            layers.append(layer + "%s" % partition)
            output_data_sizes.append(output_data_size)
            input_data_sizes.append(input_data_size)
            record_latency(layer, t, input_data_size, output_data_size, partition, layerType,imgSize)

    return intermediate


def draw(img_size):
    import matplotlib.pyplot as plt
    total_width, n = 0.8, 2
    width = total_width / n
    with open("../../latencyRes/res_alexnet_edge.txt", 'a', encoding='utf-8') as f:
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
    plt.title("alexnet edge server %s * %s" % (img_size, img_size))
    plt.savefig("../../latencyRes/plt_edge/" + "alexnet_edge_server_%s.png" % img_size, dpi=600)
    plt.legend()
    plt.show()


def res_to_csv(header=True):
    import pandas as pd
    for key in list(res.keys()):
        df = pd.DataFrame(res[key])
        if header == False:
            df.to_csv('../../latencyRes/table_edge/%s.csv' % key, encoding='utf-8', header=False, index=False, mode='a')
        else:
            df.to_csv('../../latencyRes/table_edge/%s.csv' % key, encoding='utf-8', mode='a', index=False)


def process_summaries(summaries):
    for layer in summaries:
        input_shapes.append(abs(reduce(lambda x, y: x * y, summaries[layer]["input_shape"])))
        output_shapes.append(abs(reduce(lambda x, y: x * y, summaries[layer]["output_shape"])))


if __name__ == '__main__':
    print('test partition points in AlexNet!!!')

    import json
    import torchvision.transforms as transforms
    from PIL import Image

    with open("../../data/imageNet_class_index.json", "r") as read_file:
        class_idx = json.load(read_file)
        labels = {int(key): value for key, value in class_idx.items()}

    model = lenet(pretrained=False)
    model.eval()
    if torch.cuda.is_available():
        model.cuda(0)
    imgSize = [224]
    for i in range(230, 641, 10):
        imgSize.append(i)
    for img_size in imgSize:
        print("输入图片尺寸为：", img_size)
        res = {key: {"Input Data Size(MB)": [], "Output Data Size(MB)": [], 'Input Shape': [],
                     'Output Shape': [], "Execute Time(ms)": [], "Network": [], "Platform": [], "ImgSize": []} for key in
               ['Conv2d', 'ReLU', 'MaxPool2d', 'Linear', 'Dropout']}
        input_shapes = []
        output_shapes = []
        # summaries = summary(lenets(), input_size=(3, img_size, img_size), batch_size=-1,device='cpu')
        # process_summaries(summaries)
        start_time = time.time()
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

        intermediate = img.cuda(0)
        img_time = time.time() - start_time
        # 执行features
        intermediate = execute_layers(intermediate, "features", 6, 0, img_size)

        # avgpool
        intermediate, layer, t, input_data_size, output_data_size, p = model(intermediate, partition=0,
                                                                                         layerType="avgpool")
        # print(layer, "===", t)
        # print("============")
        times.append(t)
        layers.append(layer + "%s" % 6)
        output_data_sizes.append(output_data_size)
        input_data_sizes.append(input_data_size)

        # classifier
        intermediate = execute_layers(intermediate, "classifier", 5, 0, img_size)

        prediction = torch.argmax(intermediate)

        print('img size ', img_size, labels[prediction.item()])
        print(img_time)
        print(sum(times) / 1000)
        print(sum(times) / 1000 + img_time)
        # 绘图
        draw(img_size)
        # if img_size == 224:
        #     res_to_csv(header=True)
        # else:
        #     res_to_csv(header=False)
