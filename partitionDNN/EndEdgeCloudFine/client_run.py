#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Gz
# datetime： 2021/11/4 11:03 
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Gz
# datetime： 2021/11/3 15:59
import json
import time
import argparse
import torch
import socket
from PIL import Image
from EndEdgeCloudFine.models.vgg16 import vgg16
from EndEdgeCloudFine.models.AlexNet import alexnet
import torchvision.transforms as transforms
from utils.communication_EEC import clientCommunication
from EndEdgeCloudFine.config import vgg16_config, alexnet_config
#from utils.cpu_info import get_cpu_speed, get_cpu_used, get_memory_info, gpu_util_timer, get_platform_capability

def decision(model):
    if model == "vgg16":
        return [1, 0, 1, 2, 1, 0, 1, 1, 2, 2, 0, 0, 2, 0, 2, 1, 2, 1,
                0, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 0, 0, 1, 1, 0, 1, 2, 2, 0]
    else:
        # return [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,2,2]
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

class Client:
    def __init__(self, args):
        if args.dnn_model == 'vgg16':
            self.model = vgg16()
            self.model.eval()
            self.features, self.classifiers, self.classifiers_index = vgg16_config()
        elif args.dnn_model == 'alexnet':
            self.model = alexnet()
            self.model.eval()
            self.features, self.classifiers, self.classifiers_index = alexnet_config()
        self.img, self.labels = self.dataLoad()  # 加载数据，和分类标签
        self.communication = clientCommunication(args.host_client, args.port_client)  # 监听端口
        self.start, self.sign = True, 0
        self.decision = []
        self.start_time = 0

    def client(self):
        """程序入口"""
        print("客户端启动成功！")

        while True:
            if self.start:
                # 第一次发送消息到云服务器获取决策结果
                self.get_decision_from_cloud()
                self.start = False
            else:
                conn, addr = self.communication.accept_conn()  # 接收边缘服务器返回的数据
                with conn:
                    # 获取数据
                    recv_data = self.communication.receive_msg(conn)
                    if recv_data[0] == "decision_res":  # 保存决策结果决策
                        self.start_time = time.time()
                        self.first_layer(recv_data)
                    else:
                        # 其余层
                        self.other_layers(recv_data)


    def get_decision_from_cloud(self):
        """提交客户端的计算能力，客户端到边缘服务器的上行带宽和下行带宽"""
        # capability = get_platform_capability()
        send_data = ["decision", args.dnn_model, [10, 20]]
        self.communication.send_msg(send_data, args.host_edge, args.port_edge)

    def first_layer(self, recv_data):
        """
        1）判断第一层执行位置
        2）如果在则执行
        3）不在则将原图上传到边缘服务器
        """
        self.decision = recv_data[1]
        if self.decision[0] == 0:
            # 第一层在本地执行
            intermediate, self.sign = self.model_run(self.sign, self.img)
            send_data = [self.sign + 1, intermediate, 'client']
            self.communication.send_msg(send_data, args.host_edge, args.port_edge)
        else:
            # 第一层不在本地执行
            send_data = [self.sign, self.img, 'client']
            self.communication.send_msg(send_data, args.host_edge, args.port_edge)

    def other_layers(self, recv_data):
        # 其余层
        layer_num = recv_data[0]
        data = recv_data[1]
        intermediate = torch.autograd.Variable(data)
        if layer_num >= len(self.decision):
            # 满足退出条件，进行最终的预测
            print("结束啦")
            prediction = torch.argmax(data)
            print(self.labels[prediction.item()])
            self.end_time = time.time()
            print(self.end_time-self.start_time)
        else:
            # 继续运行
            intermediate, sign = self.model_run(layer_num, intermediate)
            if sign + 1 < len(self.decision) - 1:
                data_to_edge = [sign + 1, intermediate.data, "client"]
                self.communication.send_msg(data_to_edge, args.host_edge, args.port_edge)
            else:
                prediction = torch.argmax(intermediate)
                print(self.labels[prediction.item()])
                self.end_time = time.time()
                print(self.end_time - self.start_time)


    def model_run(self, partition_point, data):
        """DNN的执行"""
        intermediate = data
        sign = 0
        for i in range(partition_point, len(self.decision)):
            if self.decision[i] == 0:
                if i < self.features:
                    # 提取特征部分
                    intermediate, layer, layer_time,input_data_size, output_data_size, partition = self.model(intermediate, partition=i, layerType="features")
                    sign = i
                    print(layer,i, "======", layer_time)
                elif i == self.features:
                    # avgpool
                    intermediate, layer, layer_time,input_data_size, output_data_size, partition = self.model(intermediate, partition=i, layerType="features")
                    print(layer,i, "======", layer_time)
                    intermediate, layer, layer_time,input_data_size, output_data_size, partition = self.model(intermediate, partition=i, layerType="avgpool")
                    sign = i
                    print(layer,i+1, "======", layer_time)
                else:
                    # 分类部分
                    intermediate, layer, layer_time,input_data_size, output_data_size, partition = self.model(intermediate, partition=self.classifiers_index[i],
                                              layerType="classifier")
                    sign = i
                    print(layer,i, "======", layer_time)
            else:
                break
        return intermediate, sign


    def dataLoad(self):
        """数据的加载"""
        min_img_size = 224
        transform_pipeline = transforms.Compose([transforms.Resize((min_img_size, min_img_size)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])
        img = Image.open('../data/Golden_Retriever_Hund_Dog.jpg')
        img = transform_pipeline(img)
        img = img.unsqueeze(0)
        with open("../data/imageNet_class_index.json", "r") as read_file:
            class_idx = json.load(read_file)
            labels = {int(key): value for key, value in class_idx.items()}
        return img, labels


if __name__ == '__main__':
    desc = 'ANS in edge server side'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dnn', dest='dnn_model', help='vgg16, alexnet', default='alexnet', type=str)
    parser.add_argument('--host_client', dest='host_client', help='Ip address', default='192.168.3.207', type=str)
    parser.add_argument('--host_edge', dest='host_edge', help='Ip address', default='192.168.3.207', type=str)
    parser.add_argument('--host_cloud', dest='host_cloud', help='Ip address', default='192.168.3.207', type=str)
    parser.add_argument('--port_client', dest='port_client', help='Ip port', default=8888, type=int)
    parser.add_argument('--port_edge', dest='port_edge', help='Ip port', default=8082, type=int)
    parser.add_argument('--port_cloud', dest='port_cloud', help='Ip port', default=8083, type=int)
    args = parser.parse_args()
    c = Client(args)
    c.client()
