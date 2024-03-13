import copy
import torch
import numpy as np
import quadprog
from qpsolvers import solve_qp

def para_diff_cal(w_global, w_local):  # 两个OrderedDict做差值
    update = copy.deepcopy(w_global)
    for key, value in w_local.items():
        if key in update:
            update[key] = update[key] - value
        else:
            update[key] = 0 - value
    return update

def sub_models(model1, model2):
    subtract = copy.deepcopy(model1)
    for key, value in subtract.items():
        with torch.no_grad():
            subtract[key] = model1[key] - model2[key]
    return subtract

# def float_mulpty_OrderedDict(float_num, weight):
#     result = copy.deepcopy(weight)
#     for key, value in result.items():
#         result[key] = float_num * value
#     return result

def float_mulpty_OrderedDict(float_num, weight):
    result = copy.deepcopy(weight)
    for key, value in result.items():
        result[key] = float_num * value
    return result

def scale_model(model, scale):
    scaled = copy.deepcopy(model)
    for key, value in scaled.items():
        scaled[key] = scale * value
    return scaled

def float_mulpty_value(float_num, weight):
    result = copy.deepcopy(weight)
    for idx, value in zip(range(len(result)), result):
        result[idx] = float_num * value
    return result

def OrderedDict_divide_float(float_num, weight):
    if float_num == 0:
        return False
    result = copy.deepcopy(weight)
    for key, value in result.items():
        result[key] = value / float_num
    return result

def normal(delta_ws):  # OrderedDict求二范数
    sum = 0.0
    for key, value in delta_ws.items():
        sum += delta_ws[key].pow(2).sum()
    return float(sum)

def norm2_model(model):
    sum_ = 0.0
    for param in model:
        sum_ += torch.norm(param) ** 2
    return sum_

def normal_test(delta_ws):  # OrderedDict求二范数
    a = copy.deepcopy(delta_ws)
    sum = []
    for key, value in a.items():
        sum.append(value)
    # print(sum)
    sum_cpu = [i.cpu() for i in sum]
    # print(sum_cpu)
    client_grads = sum_cpu[0] # shape now: (784, 26)

    for i in range(1, len(sum_cpu)):
        client_grads = np.append(client_grads, sum_cpu[i])
    return np.sum(np.square(client_grads))

def orderdict_sum(orderdict_lists):  # OrderedDict的列表求和
    result = orderdict_zeros_like(orderdict_lists[0])
    for i in orderdict_lists:
        for key, value in i.items():
            if key in result:
                result[key] += value
            else:
                result[key] =value
    return result 

def sum_models(models):
    avg = orderdict_zeros_like(models[0])
    for model in models:
        for key, value in model.items():
            avg[key] += value
    return avg

def orderdict_sum_test(orderdict_lists):  # OrderedDict的列表求和
    result = orderdict_zeros_like(orderdict_lists[0])
    for i in orderdict_lists:
        result = ms_cal(i, result)
    return result 

def orderdict_zeros_like(orderdict):
    zero = copy.deepcopy(orderdict)
    for key, value in zero.items():
        zero[key] = torch.zeros_like(value)
    return zero

def weight_sum(weight_lists):  # OrderedDict的列表求和
    result={}
    for i in weight_lists:
        for key, value in i.items():
            if key in result:
                result[key] += value
            else:
                result[key] =value
    return result

def delta_kt_sum(weight_lists):  # OrderedDict列表求和
    result = orderdict_zeros_like(weight_lists[0])
    for i in weight_lists:
        for key, value in i.items():
            if key in result:
                result[key] += value
            else:
                result[key] =value
    return result

def ms_cal(a, b):  # OrderedDict的列表求和
    result = copy.deepcopy(a)
    for key, value in b.items():
        if key in result:
            result[key] = result[key] + value
        else:
            result[key] = 0 - value
    return result

def add_models(model1, model2, alpha=1.0):
    # obtain model1 + alpha * model2 for two models of the same size
    addition = copy.deepcopy(model1)
    for key, value in addition.items():
        with torch.no_grad():
            addition[key] = model1[key] + alpha * model2[key]
    return addition

def product_models(model1, model2):
    # obtain model1 - model2 for two models of the same size
    prod = 0.0
    for key, value in model1.items():
        with torch.no_grad():
            #print('param1.data: ', param1.data, 'param2.data: ', param2.data)
            prod += torch.dot(model1[key].view(-1), model2[key].view(-1))
    return prod

def solve_centered_w(U, epsilon):
    """
        utils from FedMGDA repo
        U is a list of normalized gradients (stored as state_dict()) from n users
    """
    n = len(U)
    K = np.eye(n,dtype=float)
    for i in range(n):
        for j in range(n):
            K[i,j] = product_models(U[i], U[j])

    Q = 0.5 *(K + K.T)
    p = np.zeros(n,dtype=float)
    a = np.ones(n,dtype=float).reshape(-1,1)
    Id = np.eye(n,dtype=float)
    neg_Id = -1. * np.eye(n,dtype=float)
    lower_b = (1./n - epsilon) * np.ones(n,dtype=float)
    upper_b = (-1./n - epsilon) * np.ones(n,dtype=float)
    A = np.concatenate((a,Id,Id,neg_Id),axis=1)
    b = np.zeros(n+1)
    b[0] = 1.
    b_concat = np.concatenate((b,lower_b,upper_b))
    alpha = quadprog.solve_qp(Q,p,A,b_concat,meq=1)[0]
    #print('weights of FedMGDA: ', alpha)
    return alpha