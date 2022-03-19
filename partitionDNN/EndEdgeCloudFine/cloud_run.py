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
from utils.communication_EEC import cloudCommunication
from EndEdgeCloudFine.config import vgg16_config, alexnet_config
#from utils.cpu_info import get_platform_capability


class CloudServer:
    def __init__(self, args):
        self.communication = cloudCommunication(args.host_cloud, args.port_cloud)  # 监听端口
        self.decision = []


    def optimize(self, recv_data):
        """提交边缘服务器的计算能力，边缘服务器到云服务器的上行带宽和下行带宽"""
        if recv_data[1] == "vgg16":
            self.model = vgg16()
            self.model.eval()
            self.features, self.classifiers, self.classifiers_index = vgg16_config()
        elif recv_data[1] == "alexnet":
            self.model = alexnet()
            self.model.eval()
            self.features, self.classifiers, self.classifiers_index = alexnet_config()
        # client_capability = recv_data[2]
        # edge_capability = recv_data[4]
        # cloud_capability = get_platform_capability()
        client_edge_comm_in = recv_data[2][0]
        client_edge_comm_out = recv_data[2][1]
        edge_cloud_comm_in = recv_data[3][0]
        edge_cloud_comm_out = recv_data[3][1]
        if recv_data[1] == "vgg16":
            return [1, 0, 1, 2, 1, 0, 1, 1, 2, 2, 0, 0, 2, 0, 2, 1, 2, 1,
                0, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 0, 0, 1, 1,0, 1,2, 2, 0]
        else:
            # return [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,2,2]
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0]

    def cloud(self):
        print("云服务器启动成功！")
        while True:
            conn, addr = self.communication.accept_conn()  # 接收边缘服务器返回的数据
            with conn:
                recv_data = self.communication.receive_msg(conn)
                if recv_data[0] == "decision":  # 发送状态给云服务器，进行决策
                    self.decision = self.optimize(recv_data)
                    print(self.decision)
                    send_data = ["decision_res", self.decision]
                    self.communication.send_msg(send_data, args.host_edge, args.port_edge)
                else:
                    layer_num = recv_data[0]
                    data = recv_data[1]
                    intermediate = torch.autograd.Variable(data)
                    intermediate, sign = self.model_run(layer_num, intermediate)
                    send_data = [sign + 1, intermediate, "cloud"]
                    self.communication.send_msg(send_data, args.host_edge, args.port_edge)


    def model_run(self, partition_point, data):
        """DNN的执行"""
        intermediate = data
        sign = 0
        for i in range(partition_point, len(self.decision)):
            if self.decision[i] == 2:
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
    parser.add_argument('--dnn', dest='dnn_model', help='vgg, alexnet', default='vgg16', type=str)
    parser.add_argument('--host_client', dest='host_client', help='Ip address', default='192.168.3.207', type=str)
    parser.add_argument('--host_edge', dest='host_edge', help='Ip address', default='192.168.3.207', type=str)
    parser.add_argument('--host_cloud', dest='host_cloud', help='Ip address', default='192.168.3.207', type=str)
    parser.add_argument('--port_client', dest='port_client', help='Ip port', default=8888, type=int)
    parser.add_argument('--port_edge', dest='port_edge', help='Ip port', default=8082, type=int)
    parser.add_argument('--port_cloud', dest='port_cloud', help='Ip port', default=8083, type=int)
    args = parser.parse_args()
    c = CloudServer(args)
    c.cloud()

