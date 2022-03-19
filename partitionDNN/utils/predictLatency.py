#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Gz
# datetime： 2021/11/24 20:34
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import numpy as np
import pandas as pd
from utils.latencyPredictionModel import NeuralNetwork
import torch

def prepare_data():
    """准备数据集"""
    path = "../latencyRes/table_edge/Conv2d.csv"
    df = pd.read_csv(path)
    X = []
    Y = []
    # print(len(df['Execute Time(ms)']))
    # for i in range(j):
    #     capability = float(df['Free CPU(GHZ)'][i])*0.3+float(df['Free Memory(G)'][i])*0.3+float(df['Free Gpu(G)'][i])*0.4
    #     self.X.append([df['Data Size(MB)'][i],df['Input Data Size(MB)'][i],capability])
    #     self.Y.append(df['Execute Time(ms)'][i])
    for i in range(520):
        # if df['Network'][i] =="Alexnet":
        X.append(
            [df['Input Data Size(MB)'][i],df['Output Data Size(MB)'][i],df['Input Shape'][i],df['Output Shape'][i]])
        Y.append(df['Execute Time(ms)'][i])
    X = np.array(X)
    Y = np.array(Y)
    # print(X)
    # print(Y)
    return X, Y

model = NeuralNetwork()
x,y = prepare_data()
scaler = RobustScaler()
print(x)
X = torch.from_numpy(scaler.fit_transform(x).astype('float32'))
y = torch.from_numpy(scaler.fit_transform(y.reshape(-1, 1)).astype('float32'))
print(X[0:5])
checkpoint = torch.load("../weights/latencyModel/conv2d.pkl")
model.load_state_dict(checkpoint['state_dict'])
y_pred = model(X[0:5])
print(y_pred)
print(scaler.inverse_transform(y_pred.detach().numpy()))