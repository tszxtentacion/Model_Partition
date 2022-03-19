#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Gz
# datetime： 2021/10/27 10:05

import sklearn.datasets as datasets
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams['font.sans-serif']='SimHei'


class PredictBostonPrice:
    def __init__(self, j):
        """加载boston房价数据"""

        df = pd.read_csv('../latencyRes/table_edge/MaxPool2d_new.csv')
        self.j = len(df)
        self.X = []
        self.Y = []
        # print(len(df['Execute Time(ms)']))
        # for i in range(j):
        #     capability = float(df['Free CPU(GHZ)'][i])*0.3+float(df['Free Memory(G)'][i])*0.3+float(df['Free Gpu(G)'][i])*0.4
        #     self.X.append([df['Data Size(MB)'][i],df['Input Data Size(MB)'][i],capability])
        #     self.Y.append(df['Execute Time(ms)'][i])
        for i in range(self.j):
            # self.X.append(
            #     [df['Output Data Size(MB)'][i], df['Input Shape'][i]])
            self.X.append(
                [df['Input Data Size(MB)'][i],df['Output Data Size(MB)'][i],df['Input Shape'][i],df['Output Shape'][i]])
            self.Y.append(df['Execute Time(ms)'][i])
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        # print(self.X)
        # print(self.Y)

        # boston=datasets.load_boston()
        # self.X=boston.data
        # self.Y=boston.target
        #
        # print(self.X)
        # print(self.Y)

    def split_data(self, i):
        """将数据集划分为训练集和测试集"""
        self.i = i
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, random_state=self.i, test_size=0.3)
        # 标准化
        std = StandardScaler()
        self.x_train = std.fit_transform(X_train)
        self.x_test = std.transform(X_test)
        self.y_train = std.fit_transform(y_train.reshape(-1, 1))  # y需要转化为2维
        self.y_test = std.transform(y_test.reshape(-1, 1))

    def train(self):
        """模型训练"""
        self.LR = LinearRegression()  # 生成模型
        self.LR.fit(self.x_train, self.y_train)  # 训练模型

        self.sgdr = SGDRegressor()
        self.sgdr.fit(self.x_train, self.y_train)

    def predict_line(self):
        """模型预测"""
        plt.title("卷积层预测")
        y_pred = self.LR.predict(self.x_test)  # 模型预测
        plt.plot(self.y_test, label='真实推理时间')
        plt.plot(y_pred, label='预测推理时间')

        plt.legend()
        plt.show()

    def predict_SGD(self):
        plt.title("卷积层预测")
        sgdr_t_pred = self.sgdr.predict(self.x_test)
        score = self.sgdr.score(self.x_test,self.y_test)
        print("%s,===,%s,score:"%(self.j, self.i),score)
        # with open('./score.txt','a') as f:
        #     f.write("%s,===,%s,score:%s\n"%(self.j, self.i, score))
        MSE = mean_squared_error(self.y_test, sgdr_t_pred)
        RMSE = np.sqrt(mean_squared_error(self.y_test, sgdr_t_pred))
        print(self.sgdr.coef_)
        print(self.sgdr.intercept_)
        # print(MSE)
        # print(RMSE)
        # plt.plot(self.y_test, label='真实推理时间')
        # plt.plot(sgdr_t_pred, label='预测推理时间')
        # plt.scatter(self.y_test, sgdr_t_pred)
        # plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--')
        # plt.legend()
        # plt.show()

if __name__ == '__main__':
    PBP = PredictBostonPrice(10)
    PBP.split_data(10)
    # PBP.show_data()
    PBP.train()
    # PBP.predict_line()
    PBP.predict_SGD()
    # for j in range(100,910):
    #     PBP = PredictBostonPrice(j)
    #     for i in range(100):
    #         PBP.split_data(i)
    #         # PBP.show_data()
    #         PBP.train()
    #         # PBP.predict_line()
    #         PBP.predict_SGD()
    # PBP = PredictBostonPrice(113)
    # PBP.split_data(65)
    # # PBP.show_data()
    # PBP.train()
    # # PBP.predict_line()
    # PBP.predict_SGD()

# Conv 391 9 0.8500943215936602
# MaxPool 328 30 0.9190316061491058
# Line  338 89 0.8523548218480699
# Relu 697 26 0.8745042569790353
# Droupout 113 65 0.9516463352299862