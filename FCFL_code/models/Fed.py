#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):   # main中将w_locals赋给w，即worker计算出的权值
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():  # 对于每个参与的设备：
        for i in range(1, len(w)):  # 对本地更新进行聚合
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
