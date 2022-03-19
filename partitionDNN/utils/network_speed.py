#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Gz
# datetime： 2021/11/22 10:28 

def fourG(imgSize):
    download = imgSize * 1024 / (43.23 * 1024 * 1024 /8)
    upload = imgSize * 1024 / (17.61 *1024 * 1024 /8)
    return download, upload

def fiveG(imgSize):
    download = imgSize * 1024 / (237.21 *1024 * 1024 /8)
    upload = imgSize * 1024 / (40.34 *1024 * 1024 /8)
    return download, upload

def wifiG(imgSize):
    download = imgSize * 1024 / (48.45 *1024 * 1024 /8)
    upload = imgSize * 1024 / (24.86 *1024 * 1024 /8)
    return download, upload

if __name__ == '__main__':
    print(fourG(152))
    print(fiveG(152))
    print(wifiG(152))