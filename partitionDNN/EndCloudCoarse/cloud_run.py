#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Gz
# datetime： 2021/11/4 11:03
# !/usr/bin/env python
import argparse
import torch
from EndEdgeCloudFine.models.vgg16 import vgg16
from utils.communication_EEC import edgeCommunication
from EndEdgeCloudFine.config import vgg16 as vgg16_config

features = vgg16_config['features']
classifiers = vgg16_config['classifier']
classifiers_index = {31: 0, 32: 1, 33: 2, 34: 3, 35: 4, 36: 5, 37: 6}


def cloud(args):
    print("参数为：", args)
    if args.dnn_model == 'vgg':
        model = vgg16()
        model.eval()
    else:
        model = vgg16()

    decision = [0, 0, 1, 2, 1, 0, 1, 1, 2, 2, 0, 0, 2, 0, 2, 1, 2, 1, 0, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 0, 0, 1, 1, 0, 1,
                2, 2, 1]

    communication = edgeCommunication(args.host_cloud, args.port_cloud)
    while True:
        conn, addr = communication.accept_conn()
        with conn:
            # 接收数据
            recv_data = communication.receive_msg(conn)
            print("从%s接收数据成功" % recv_data[2])
            partition_point = recv_data[0]
            data = recv_data[1]
            intermediate = torch.autograd.Variable(data)
            intermediate, sign = model_run(partition_point, intermediate, decision, model)
            send_data = [sign + 1, intermediate, "cloud"]
            communication.send_msg(send_data, args.host_edge, args.port_edge)


def vgg16_():
    from EndEdgeCloudFine.config import vgg16 as vgg16_config

    features = vgg16_config['features']
    classifiers = vgg16_config['classifier']
    classifiers_index = {31: 0, 32: 1, 33: 2, 34: 3, 35: 4, 36: 5, 37: 6}
    return features, classifiers, classifiers_index


def model_run(partition_point, data, decision, model):
    intermediate = data
    sign = 0
    for i in range(partition_point, len(decision)):
        if decision[i] == 2:
            if i < features:
                intermediate = model(intermediate, partition=i, layerType="features")
                sign = i
            elif i == features:
                intermediate = model(intermediate, partition=i, layerType="features")
                intermediate = model(intermediate, partition=i, layerType="avgpool")
                sign = i
            else:
                intermediate = model(intermediate, partition=classifiers_index[i], layerType="classifier")
                sign = i
            print(i)
        else:
            break
    return intermediate, sign


if __name__ == '__main__':
    desc = 'ANS in edge server side'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dnn', dest='dnn_model', help='vgg, alexnet', default='vgg', type=str)
    parser.add_argument('--host_client', dest='host_client', help='Ip address', default='127.0.0.1', type=str)
    parser.add_argument('--host_edge', dest='host_edge', help='Ip address', default='127.0.0.1', type=str)
    parser.add_argument('--host_cloud', dest='host_cloud', help='Ip address', default='127.0.0.1', type=str)
    parser.add_argument('--port_client', dest='port_client', help='Ip port', default=8080, type=int)
    parser.add_argument('--port_edge', dest='port_edge', help='Ip port', default=8081, type=int)
    parser.add_argument('--port_cloud', dest='port_cloud', help='Ip port', default=8082, type=int)
    args = parser.parse_args()
    print(args)
    cloud(args)
