#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np
from utils.weight_cal import weight_sum, para_diff_cal, delta_kt_sum, float_mulpty_OrderedDict, OrderedDict_divide_float, orderdict_sum, float_mulpty_value, scale_model, sub_models, sum_models
import collections

def q_FedAvg(global_model, delta_ks, hks):   # main中将w_locals赋给w，即worker计算出的权值 
    new_model = copy.deepcopy(global_model)
    update = []
    sum_h_kt = np.sum(np.asarray(hks))

    for delta_k in delta_ks:
        update.append(float_mulpty_OrderedDict(1.0/sum_h_kt, delta_k))

    values = orderdict_sum(update)
    with torch.no_grad():
        model = para_diff_cal(new_model, values)
    return model

    
def q_aggregate(global_model, delta_ks, hks):
    sum_h_kt = np.sum(np.asarray(hks))
    new_model = sub_models(global_model, scale_model(sum_models(delta_ks), 1.0 / sum_h_kt))
    return new_model


# def FedAvg(w):   # main中将w_locals赋给w，即worker计算出的权值
#     w_avg = copy.deepcopy(w[0])
#     for k in w_avg.keys():  # 对于每个参与的设备：
#         for i in range(1, len(w)):  # 对本地更新进行聚合
#             w_avg[k] += w[i][k]
#         w_avg[k] = torch.div(w_avg[k], len(w))
#     return w_avg


def FedAvg(w):   # main中将w_locals赋给w，即worker计算出的权值
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():  # 对于每个参与的设备：
        for i in range(1, len(w)):  # 对本地更新进行聚合
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def weight_agg(model_list, weight_list):
    new_model_key = model_list[0].keys()
    new_model_value = [float_mulpty_OrderedDict(j, i) for i, j in zip(model_list, weight_list)]
    # new_model = dict(zip(new_model_key, new_model_value))
    new_model_value = weight_sum(new_model_value)
    # print(new_model_value)
    # print(collections.OrderedDict(new_model_value))
    return collections.OrderedDict(new_model_value)

def acc_global_estimate(acc_list, weight_list):
    result = 0.0
    num = len(acc_list)
    for i in range(num):
        result = result + acc_list[i] * weight_list[i]
    return result

def margin_glob_model(model_list, weight_list):
    margin_glob_model = []
    length = len(model_list)
    for i in range(length):
        del_model_list = copy.deepcopy(model_list)
        del del_model_list[i]
        del_weight_list = copy.deepcopy(weight_list)
        del del_weight_list[i]
        new_model_value = [float_mulpty_OrderedDict(j, i) for i, j in zip(del_model_list, del_weight_list)]
        new_model_value = weight_sum(new_model_value)
        margin_glob_model.append(collections.OrderedDict(new_model_value))
    return margin_glob_model


