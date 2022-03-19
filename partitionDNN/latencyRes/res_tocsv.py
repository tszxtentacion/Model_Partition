#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Gz
# datetime： 2021/11/22 20:58
import json
import pandas as pd
import matplotlib.pyplot as plt
import ast
conv2 = []
def load_res():
    with open("./res_alexnet_edge.txt", 'r') as f:
        lines = f.readlines()
        for j in range(224,641):
            for i, line in enumerate(lines):
                if line == '%s\n'%j:
                    conv2.append(ast.literal_eval(lines[i+2])[4])
                    # return ast.literal_eval(lines[i+1]), ast.literal_eval(lines[i+2]), ast.literal_eval(lines[i+4])

# layers, latency, output_size = load_res()
load_res()
print(conv2)
plt.plot([i for i in range(len(conv2))], conv2)
plt.show()
# df = pd.DataFrame({"layers": layers, "latency":latency,"output_size":output_size})
# df.to_csv("./res_224.csv", encoding="utf-8")