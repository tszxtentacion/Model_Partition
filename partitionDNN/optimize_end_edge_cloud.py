#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Gz
# datetime： 2021/11/25 9:48
import data_config
import matplotlib.pyplot as plt

# 用时间来衡量通信开销
def communication_cost(mobile_index, edge_index, cloud_index, output_data):
    point_data_1 = output_data[mobile_index[-1]]        # 从端设备离开时的输出
    # print("point1_data: ",point_data_1)
    # print(point_data_1)
    if edge_index == [] and cloud_index == []:
        # 全在端
        c_t = 0
    elif edge_index == []:
        # 边缘服务器上不执行
        # print((point_data_1 * 1024) / edge_cloud)
        # print("1")
        # mobile_edge：从端到边上的传输速度； edge_cloud：从边到云上的传输速度（比特数/传输速度）
        c_t = ((point_data_1 * 1024) / mobile_edge + (point_data_1 * 1024) / edge_cloud) * 1000  # ms
    elif cloud_index == []:
        # 云上不执行
        # print("2")
        c_t = ((point_data_1 * 1024) / mobile_edge) * 1000
    else:
        # print("3")
        # print(edge_index[-1])
        point_data_2 = output_data[edge_index[-1]]      # 从边上离开时的输出
        # print("point2_data: ", point_data_2)
        c_t = ((point_data_1 * 1024) / mobile_edge + (point_data_2 * 1024) / edge_cloud) * 1000
    return c_t


def computing_latency(mobile_index, edge_index, cloud_index, time_mobile, time_edge, time_cloud):
    # 在端上计算
    if mobile_index == []:
        t_mobile = 0
    else:
        t_mobile = sum([time_mobile[i] for i in mobile_index])
    # 在边上计算
    if edge_index == []:
        t_edge = 0
    else:
        t_edge = sum([time_edge[i] for i in edge_index])
    # 在云上计算
    if cloud_index == []:
        t_cloud = 0
    else:
        t_cloud = sum([time_cloud[i] for i in cloud_index])
    # 求和
    return t_mobile + t_edge + t_cloud


def optimize(time_mobile, time_edge, time_cloud, output_data):
    """寻找最佳切割点"""
    latency = []    # 一种方案的总耗时（通信+计算）
    location = []   # 分配方案
    # point1，point2为端-边和边-云之间的分割点，用for进行两点组合的遍历
    for point_1 in range(0, len(time_mobile)):  
        # 客户端上任务id（范围）
        mobile_index = [i for i in range(len(time_mobile[0:point_1 + 1]))]
        for point_2 in range(point_1 + 1, len(time_mobile) + 1):
            # 边缘服务器上任务id（范围）
            edge_index = [i for i in range(point_1 + 1, point_2)]
            # 云服务器上任务id（范围）
            cloud_index = [i for i in range(point_2, len(time_mobile))]
            # 计算 通信代价
            t_comm = communication_cost(mobile_index, edge_index, cloud_index, output_data)
            # 计算 计算代价
            t_computing = computing_latency(mobile_index, edge_index, cloud_index, time_mobile, time_edge, time_cloud)
            latency.append(t_comm + t_computing)
            # 将该分割方案放入列表
            location.append([mobile_index, edge_index, cloud_index])

    min_index = latency.index(min(latency))
    print("end-edge-cloud: ",min(latency))
    print(location[min_index])
    return location[min_index],min(latency)


if __name__ == '__main__':
    # nets = [1.56, 18.66, 36.38, 39.19]
    nets = [i for i in range(150, 1001, 25)]
    print(nets)
    nets = [150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
    latencies = []
    for net in nets:
        print("net: ", net)
        mobile_edge = 39.19 * 1024 / 8  # 17.61*1024/8 4G  5G 40.34*1024/8 # 200  # wifi 24.86
        edge_cloud = net * 1024 / 8

        time_mobile, time_edge, time_cloud, output_data =data_config.alexNet_data_224()
        index, latency = optimize(time_mobile, time_edge, time_cloud, output_data)
        latencies.append(latency)
        print("=====================")
    print(latencies)
    import pandas as pd
    a = pd.DataFrame({"k":latencies})
    a.to_csv("./latencies_cloud.csv", encoding="utf-8")
    plt.plot(nets, latencies, label='Layer Latency', marker='o')
    plt.xticks(rotation=270)
    plt.title("alexnet edge server 224 * 224")
    # plt.savefig("../../latencyRes/plt_edge/" + "alexnet_edge_server_%s.png" % img_size, dpi=600)
    plt.legend()
    plt.show()


    # mobile_edge = 39.19 * 1024 / 8  # 17.61*1024/8 4G  5G 40.34*1024/8 # 200  # wifi 24.86
    # edge_cloud = 175.51 * 1024 / 8
    #
    # print("size: ", 224)
    # time_mobile, time_edge, time_cloud, output_data =data_config.vgg11_data_640()
    # # time_mobile, time_edge, time_cloud, output_data = vgg16_data_360()
    # # print("mobile:", sum(time_mobile) / 1000)
    # # print("edge:", sum(time_edge_1) / 1000)
    # # print("cloud:", sum(time_cloud) / 1000)
    # optimize(time_mobile, time_edge, time_cloud, output_data)
    # print("=====================")
