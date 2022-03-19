#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Gz
# datetime： 2021/11/3 15:59
import json
import time
import argparse
import torch
from PIL import Image
from EndCloudCoarse.models.vgg16 import vgg16
import torchvision.transforms as transforms
from utils.communication_EnC import clientCommunication


def client(args):
    print("参数为：", args)
    if args.dnn_model == 'vgg':
        model = vgg16()
        model.eval()
    else:
        model = vgg16()

    model.cuda()
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

    communication = clientCommunication(args.host, args.port)
    for partition in range(23):
        with torch.no_grad():
            # 客户端执行
            start_client_time = time.time()
            intermediate = model(img.cuda(), server=False, partition=partition)
            data_to_server = [partition, intermediate.data]
            del intermediate
            communication.send_msg(data_to_server)
            end_client_time = time.time()

            # 云端执行
            start_cloud_time = time.time()
            prediction = communication.receive_msg()
            prediction = torch.argmax(prediction)
            end_cloud_time = time.time()

            print('partition point ', partition, labels[prediction.item()], (end_cloud_time-start_cloud_time+end_client_time-start_client_time))


if __name__ == '__main__':
    desc = 'ANS in edge server side'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dnn', dest='dnn_model', help='vgg, alexnet', default='vgg', type=str)
    parser.add_argument('--host', dest='host', help='Ip address', default='192.168.1.121', type=str)
    parser.add_argument('--port', dest='port', help='Ip port', default=8080, type=int)
    args = parser.parse_args()
    client(args)
