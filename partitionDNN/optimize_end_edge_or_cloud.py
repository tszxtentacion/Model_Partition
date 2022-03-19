#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Gz
# datetime： 2021/11/25 9:48
import data_config


def communication_cost(mobile_index, other_index, output_data, server_type):
    point_data_1 = output_data[mobile_index[-1]]
    if other_index == []:
        # 全在端
        c_t = 0
    elif server_type == "edge":
        # 边缘服务器上不执行
        c_t = ((point_data_1 * 1024) / mobile_edge) * 1000
    elif server_type == "cloud":
        # 云上不执行
        c_t = ((point_data_1 * 1024) / mobile_edge + (point_data_1 * 1024) / edge_cloud) * 1000
    else:
        c_t = 0
    return c_t


def computing_latency(mobile_index, edge_index, cloud_index, time_mobile, time_edge, time_cloud):
    if mobile_index == []:
        t_mobile = 0
    else:
        t_mobile = sum([time_mobile[i] for i in mobile_index])

    if edge_index == []:
        t_edge = 0
    else:
        t_edge = sum([time_edge[i] for i in edge_index])

    if cloud_index == []:
        t_cloud = 0
    else:
        t_cloud = sum([time_cloud[i] for i in cloud_index])
    return t_mobile + t_edge + t_cloud


def optimize(time_mobile, time_edge, time_cloud, output_data):
    """寻找最佳切割点"""
    latency_edge = []
    location_edge = []
    latency_cloud = []
    location_cloud = []
    t_edge_comms = []
    t_cloud_comms = []
    for point_1 in range(0, len(time_mobile)):  # 切割点1
        # 客户端上任务id
        mobile_index = [i for i in range(len(time_mobile[0:point_1 + 1]))]
        edge_index = [i for i in range(point_1 + 1, len(time_mobile))]
        cloud_index = [i for i in range(point_1 + 1, len(time_mobile))]
        # 计算 通信代价
        t_comm_edge = communication_cost(mobile_index, edge_index, output_data, "edge")
        t_comm_cloud = communication_cost(mobile_index, cloud_index, output_data, "cloud")
        t_edge_comms.append(t_comm_edge)
        t_cloud_comms.append(t_comm_cloud)
        # 计算 计算代价
        t_computing_edge = computing_latency(mobile_index, edge_index, [], time_mobile, time_edge, time_cloud)
        t_computing_cloud = computing_latency(mobile_index, [], cloud_index, time_mobile, time_edge, time_cloud)
        latency_edge.append(t_comm_edge + t_computing_edge)
        location_edge.append([mobile_index, edge_index])
        latency_cloud.append(t_comm_cloud + t_computing_cloud)
        location_cloud.append([mobile_index, cloud_index])

    min_index_edge = latency_edge.index(min(latency_edge))
    min_index_cloud = latency_cloud.index(min(latency_cloud))
    # min_comm_edge = t_edge_comms[min_index_edge]
    # min_comm_cloud = t_cloud_comms[min_index_cloud]
    # print(min_comm_edge)
    # print(min_comm_cloud)
    print("end-edge:", min(latency_edge))
    print("edge best partition point:", location_edge[min_index_edge])

    print("end-cloud:", min(latency_cloud))
    print("end-cloud best partition point:", location_cloud[min_index_cloud])


if __name__ == '__main__':
    # nets = [1.56, 18.66, 36.38, 39.19]
    #
    # for net in nets:
    #     print("net: ", net)
    #     mobile_edge = net * 1024 / 8  # 17.61*1024/8 4G  5G 40.34*1024/8 # 200  # wifi 24.86
    #     edge_cloud = 175.51 * 1024 / 8
    #     time_mobile, time_edge, time_cloud, output_data =  data_config.alexNet_data_224()
    #     # print(360*360*3 / 1024)
    #     # time_mobile, time_edge, time_cloud, output_data = vgg16_data()
    #     print("end-only:", sum(time_mobile))
    #     print("edge-only:", sum(time_edge) + 147 / mobile_edge * 1000)
    #     print("cloud-only:", sum(time_cloud) + (147 / mobile_edge + 147 / edge_cloud) * 1000)
    #     optimize(time_mobile, time_edge, time_cloud, output_data)
    #     print("=============================")
    mobile_edge = 39.19 * 1024 / 8  # 17.61*1024/8 4G  5G 40.34*1024/8 # 200  # wifi 24.86
    edge_cloud = 175.51 * 1024 / 8
    time_mobile, time_edge, time_cloud, output_data = data_config.vgg16_data_520()
    # print(360*360*3 / 1024)
    # time_mobile, time_edge, time_cloud, output_data = vgg16_data()
    img_size = (520*520*3/1024)
    print("end-only:", sum(time_mobile))
    print("edge-only:", sum(time_edge) + img_size / mobile_edge * 1000)
    print("cloud-only:", sum(time_cloud) + (img_size / mobile_edge + img_size / edge_cloud) * 1000)
    optimize(time_mobile, time_edge, time_cloud, output_data)
    print("=============================")