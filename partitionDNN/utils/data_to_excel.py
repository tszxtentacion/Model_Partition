#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Gz
# datetime： 2021/11/7 10:54 
import pandas as pd

def data_to_excel(df, path):
    df = pd.DataFrame(df)
    df.to_excel()