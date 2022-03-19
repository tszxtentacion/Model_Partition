#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Gz
# datetime： 2021/11/14 16:42 
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def prepare_data():
    """准备数据集"""
    path = "../latencyRes/table_edge/Conv2d.csv"
    df = pd.read_csv(path)
    X = []
    Y = []
    for i in range(300): #conv: 520
        # X.append(
        #     [df['Output Data Size(MB)'][i],df['Input Shape'][i]])
        X.append(
            [df['Input Data Size(MB)'][i],df['Output Data Size(MB)'][i],df['Input Shape'][i],df['Output Shape'][i]])
        Y.append(df['Execute Time(ms)'][i])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 16)
        # self.layer4 = nn.Linear(128, 256)
        # self.layer5 = nn.Linear(256, 128)
        # self.layer6 = nn.Linear(128, 32)
        self.layer4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        # x = torch.relu(self.layer4(x))
        # x = torch.relu(self.layer5(x))
        # x = torch.relu(self.layer6(x))
        x = self.layer4(x)
        return x

if __name__ == '__main__':
    X, y = prepare_data()
    scaler = RobustScaler()

    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    X_train = torch.from_numpy(X_train.astype('float32'))
    X_test = torch.from_numpy(X_test.astype('float32'))
    y_train = torch.from_numpy(y_train.reshape(-1, 1).astype('float32'))
    y_test = torch.from_numpy(y_test.reshape(-1, 1).astype('float32'))

    dataset_train = torch.utils.data.TensorDataset(X_train, y_train)
    dataset_train = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)
    model = NeuralNetwork()

    loss_obj = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    losses = []
    acces = []
    for i in tqdm(range(1000)):
        optimizer.zero_grad()
        loss_temp = []
        acc_temp = []
        for X, y in dataset_train:
            y_pred = model(X)
            loss = loss_obj(y_pred, y)
            acc_temp.append(r2_score(y.detach().numpy(), y_pred.detach().numpy()))
            loss_temp.append(np.float(loss))
            loss.backward()
            optimizer.step()
        losses.append(np.min(loss_temp))
        print(np.max(acc_temp))
        acces.append(np.max(acc_temp))

    # torch.save({'state_dict': model.state_dict()}, "../weights/latencyMobileModel/MaxPool2d_mobile.pkl")

    import matplotlib.pyplot as plt
    import numpy as np

    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.plot(x, acces)
    plt.show()
    df = pd.DataFrame({"losses":acces})
    df.to_csv("../weights/latencyMobileModel/MaxPool2d_acc_edge.csv", encoding='utf-8')
    # df.to_csv("../weights/latencyMobileModel/MaxPool2d_mobile.csv", encoding='utf-8')

    # checkpoint = torch.load("../weights/latencyModel/conv2d.pkl")
    # model.load_state_dict(checkpoint['state_dict'])
    # x = torch.from_numpy(scaler.fit_transform(np.array([[0.574272156,0.738578796,150528,193600]])).astype('float32'))
    y_pred = model(X_test)
    # print(scaler.inverse_transform(y_test.detach().numpy()))
    # print(scaler.inverse_transform(y_pred.detach().numpy()))
    print(r2_score(y_test.detach().numpy(), y_pred.detach().numpy()))
