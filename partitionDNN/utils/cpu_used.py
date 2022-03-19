#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Gz
# datetime： 2021/11/20 8:42 
"""
runing.py -c 2 -t 0.01 -m 1000
-c cpu核数，不加-c参数为最大核数
-t cpu运算频率时间，间隔，越小占用越高
-m 内存占用，1000MB
"""
import time
import argparse
from multiprocessing import Process
from multiprocessing import cpu_count
import math
# import tempfile

# # 屏蔽 Traceback信息
# sys.stderr=tempfile.TemporaryFile()


def exec_func(bt):

    while True:
        for i in range(0, 9600000):
            pass
        time.sleep(bt)



if __name__ == "__main__":
    "cpu_count()"
    parse = argparse.ArgumentParser(description='runing')

    parse.add_argument(
        "-c",
        "--count",
        default= 6,
        help='cpu count'
        )

    parse.add_argument(
        "-t",
        "--time",
        default= 0.1,
        help='cpu time'
        )
    parse.add_argument(
        "-m",
        "--memory",
        default= 1000,
        help='memory'
        )


    args = parse.parse_args()

    cpu_logical_count = int(args.count)

    cpu_sleep_time = args.time

    memory_used_mb = int(args.memory)

    try:
        cpu_sleep_time = int(args.time)
    except ValueError:
        try:
            cpu_sleep_time = float(args.time)
        except ValueError as ex:
            raise ValueError(ex)

    _doc = """
            runing.py -c 2 -t 0.01 -m 1000
            -c 指定cpu核数，不加-c参数默认为当前cpu最大核数
            -t cpu运算频率时间，间隔，越小占用越高
            -m 内存占用，1000MB

            CPU使用率需要手动增加减少-t参数来达到，预期使用率。
           """
    print("\n====================使用说明=========================")
    print("{}".format(_doc))
    print("\n====================================================")
    print('\n当前占用CPU核数:{}'.format(cpu_logical_count))
    print('\n内存预计占用:{}MB'.format(memory_used_mb))
    print('\n资源浪费中......')

    try:
        # 内存占用
        s = ' ' * (memory_used_mb * 1024 * 1024)
    except MemoryError:
        print("剩余内存不足，内存有溢出......")

    try:

        p = Process(target=exec_func, args=("bt",))

        ps_list = []

        for i in range(0, cpu_logical_count):
            ps_list.append(Process(target=exec_func, args=(cpu_sleep_time,)))

        for p in ps_list:
            p.start()

        for p in ps_list:
            p.join()

    except KeyboardInterrupt:
        print("资源浪费结束!")