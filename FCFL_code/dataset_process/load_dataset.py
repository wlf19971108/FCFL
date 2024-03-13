import os
import struct
import numpy as np
import pickle
import json
import logging
import numpy as np
import torch
import torch.utils.data as data


def load_mnist(path,kind='train'):
    """Load MNIST data from path"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')  # kind决定选定测试集还是训练集
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')  # train为训练集；t10k为测试集
    with open(labels_path,'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        # 使用struct.unpack方法读取前两个数据。lbpath.read(8)表示一次从文件中读取8个字节
        # 这样读到的前两个数据分别是magic number（2049）和样本个数（60000）
        labels = np.fromfile(lbpath,dtype=np.uint8)
        # 读取标签，标签的数值在0~9之间
    with open(images_path,'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",imgpath.read(16))
        # 使用struct.unpack方法读取前四个数据。lbpath.read(16)表示一次从文件中读取16个字节
        # 四个数据分别是magic number（2051）、图像数量（60000）、图像的高rows（28）、图像的宽columns（28）
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
        # 设置图像形状，高度宽度均为28，通道数为1
    return images,labels

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar_train(path):
    dict_1 = unpickle(f'{path}/data_batch_1')
    dict_2 = unpickle(f'{path}/data_batch_2')
    dict_3 = unpickle(f'{path}/data_batch_3')
    dict_4 = unpickle(f'{path}/data_batch_4')
    dict_5 = unpickle(f'{path}/data_batch_5')
    label_1 = np.array(dict_1[b'labels'])
    image_1 = dict_1[b'data']
    label_2 = np.array(dict_2[b'labels'])
    image_2 = dict_2[b'data']
    label_3 = np.array(dict_3[b'labels'])
    image_3 = dict_3[b'data']
    label_4 = np.array(dict_4[b'labels'])
    image_4 = dict_4[b'data']
    label_5 = np.array(dict_5[b'labels'])
    image_5 = dict_5[b'data']
    # image_1 = image_1.astype('float32')
    # image_1 /= 255
    # image_2 = image_2.astype('float32')
    # image_2 /= 255
    # image_3 = image_3.astype('float32')
    # image_3 /= 255
    # image_4 = image_4.astype('float32')
    # image_4 /= 255
    # image_5 = image_5.astype('float32')
    # image_5 /= 255
    x_train = np.concatenate((image_1, image_2, image_3,image_4, image_5), axis = 0)
    y_train = np.concatenate((label_1, label_2, label_3, label_4, label_5), axis = 0)
    return x_train, y_train

def load_cifar_test(path):
    dict_test = unpickle(f'{path}/test_batch')
    label_test = np.array(dict_test[b'labels'])
    image_test = dict_test[b'data']
    # image_test = image_test.astype('float32')
    # image_test /= 255
    return image_test, label_test

def load_cifar(path):
    x_train, y_train = load_cifar_train(path)
    x_test, y_test = load_cifar_test(path)
    return x_train, y_train, x_test, y_test

def load_fmnist(path,kind='train'):
    """Load Fashion MNIST data from path"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')  # kind决定选定测试集还是训练集
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')  # train为训练集；t10k为测试集
    with open(labels_path,'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        # 使用struct.unpack方法读取前两个数据。lbpath.read(8)表示一次从文件中读取8个字节
        # 这样读到的前两个数据分别是magic number（2049）和样本个数（60000）
        labels = np.fromfile(lbpath,dtype=np.uint8)
        # 读取标签，标签的数值在0~9之间
    with open(images_path,'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",imgpath.read(16))
        # 使用struct.unpack方法读取前四个数据。lbpath.read(16)表示一次从文件中读取16个字节
        # 四个数据分别是magic number（2051）、图像数量（60000）、图像的高rows（28）、图像的宽columns（28）
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
        # 设置图像形状，高度宽度均为28，通道数为1
    return images,labels

def load_synthetic_train():
    train_file_path = "d:\PythonCode\phd\FCFL\dataset\synthetic_iid\data\\train\mytrain.json"

    with open(train_file_path, "r") as train_f:
        train_data = json.load(train_f)   # 读取json文件
        # print(train_data)
        client_ids_train = train_data["users"]
        # print(client_ids_train)
        client_id = []
        for i in client_ids_train:
            id_int = int(i[5:])
            client_id.append(id_int)
        # print(client_id)
        client_num = len(train_data["users"])
        # print(client_num)

        image = [[] for _ in range(client_num)]
        label = [[] for _ in range(client_num)]
        for i in range(len(client_ids_train)):
            image[i] = torch.tensor(train_data["user_data"][client_ids_train[i]]["x"])
            label[i] = torch.tensor(train_data["user_data"][client_ids_train[i]]["y"])
        # print(len(image[0]))
        # print(image[0])
        # print(label[0])
        return image, label
    
def load_synthetic_test():
    test_file_path = "d:\PythonCode\phd\FCFL\dataset\synthetic_1_1\data\\test\mytest.json"

    with open(test_file_path, "r") as test_f:
        test_data = json.load(test_f)   # 读取json文件
        # print(train_data)
        client_ids_train = test_data["users"]
        # print(client_ids_train)
        client_id = []
        for i in client_ids_train:
            id_int = int(i[5:])
            client_id.append(id_int)
        # print(client_id)
        client_num = len(test_data["users"])
        # print(client_num)

        image = [[] for _ in range(client_num)]
        label = [[] for _ in range(client_num)]
        for i in range(len(client_ids_train)):
            image[i] = torch.tensor(test_data["user_data"][client_ids_train[i]]["x"])
            label[i] = torch.tensor(test_data["user_data"][client_ids_train[i]]["y"])
        # print(len(image[0]))
        # print(image[0])
        # print(label[0])
        return image, label
        
def synthetic_dict_users_train():
    train_file_path = "d:\PythonCode\phd\FCFL\dataset\synthetic_iid\data\\train\mytrain.json"

    with open(train_file_path, "r") as train_f:
        train_data = json.load(train_f)   # 读取json文件
        # print(train_data)
        client_ids_train = train_data["users"]
        # print(client_ids_train)
        client_id = []
        for i in client_ids_train:
            id_int = int(i[5:])
            client_id.append(id_int)
        # print(client_id)
        client_num = len(train_data["users"])
        # print(client_num)

        image = [[] for _ in range(client_num)]
        label = [[] for _ in range(client_num)]
        for i in range(len(client_ids_train)):
            image[i] = torch.tensor(train_data["user_data"][client_ids_train[i]]["x"])
            label[i] = torch.tensor(train_data["user_data"][client_ids_train[i]]["y"])
        # print(len(image[0]))
        # print(image[0])
        # print(label[0])
        return image, label


def test_train_allocation(x_train, y_train, x_test, y_test, ratio):
    x_train_new = np.empty_like(x_train)
    y_train_new = np.empty_like(y_train)
    x_test_new = np.empty_like(x_test)
    y_test_new = np.empty_like(y_test)
    # print(y_test_new)

    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    num = len(x_all)
    all_idxs = list(range(num))
    test_select = np.random.choice(all_idxs, int(num*ratio), replace=False)  # 选定为测试集的列表
    train_select = list(set(all_idxs) - set(test_select))
    # print(test_select)

    for i,j in zip(range(int(num*ratio)),test_select):
        x_test_new[i] = x_all[j]
        y_test_new[i] = y_all[j]
    for i,j in zip(range(num - int(num*ratio)),train_select):
        x_train_new[i] = x_all[j]
        y_train_new[i] = y_all[j]
    return x_train_new, x_test_new, y_train_new, y_test_new

def test_train_mixture(x_train, y_train, x_test, y_test):
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    return x_all, y_all

def subtract_lists(list1, list2):
    return [element for element in list1 if element not in list2]

def server_client_split(data, label, ratio):
    num = int(len(label) * ratio)
    client_index = random.sample(range(len(label)), num)
    server_index = subtract_lists(range(len(label)), client_index)
    server_data = np.array([data[i] for i in server_index])
    server_label = np.array([label[i] for i in server_index])
    client_data = np.array([data[i] for i in client_index])
    client_label = np.array([label[i] for i in client_index])

    # print(len(server_data))
    # print(server_data.shape)
    # print(server_label)
    # print(server_data.shape)
    # print(server_label.shape)

    return server_data, server_label, client_data, client_label

from collections import defaultdict
import random
def test_train_split(dict_users, ratio):
    dict_users_train = {}
    dict_users_test = {}
    for i in dict_users:
        num = max(int(len(dict_users[i]) * ratio), 1)
        if len(dict_users[i]) == 1:
            train_sample = random.sample(list(dict_users[i]), num)
            dict_users_train[i] = dict_users_test[i] = train_sample
        else:
            train_sample = random.sample(list(dict_users[i]), num)
            test_sample = subtract_lists(dict_users[i], train_sample)
            dict_users_train[i] = train_sample
            dict_users_test[i] = test_sample
    return dict_users_train, dict_users_test

def train_test_val_split(dict_users):
    dict_users_train = {}
    dict_users_test = {}
    dict_users_val = {}
    for i in dict_users:
        train_num = max(int(len(dict_users[i]) * 0.8), 1)
        test_num = max(int(len(dict_users[i]) * 0.1), 1)
        val_num = len(dict_users[i])-train_num-test_num
        if len(dict_users[i]) == 1:
            train_sample = random.sample(list(dict_users[i]), train_num)
            dict_users_train[i] = dict_users_test[i] = dict_users_val[i] = train_sample
        else:
            train_sample = random.sample(list(dict_users[i]), train_num)
            remainder = subtract_lists(dict_users[i], train_sample)
            test_sample = random.sample(list(remainder), test_num)
            val_sample = subtract_lists(remainder, test_sample)
            dict_users_train[i] = train_sample
            dict_users_test[i] = test_sample
            dict_users_val[i] = val_sample
    return dict_users_train, dict_users_test, dict_users_val

def mnist_iid(x_train, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:  数据集,dataset_train
    :param num_users:  客户端数量,args.num_users
    :return: dict of image index 实例：{0: {0, 9}, 1: {5, 7}, 2: {2, 4}, 3: {3, 6}, 4: {8, 1}}  五个客户端,每个客户端有两个数据集的index
    """
    num_items = int(len(x_train)/num_users)  # 每个客户端的数据量
    dict_users, all_idxs = {}, [i for i in range(len(x_train))]
    for i in range(num_users):
        # set()函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False)) # 从all_idxs中采样，生成数据量无重复的一个字典，作为dict_users的第i个元素
        all_idxs = list(set(all_idxs) - dict_users[i]) # 取差集，删去已经被分配好的数据，直至每个用户都被分配了等量的iid数据
    return dict_users


def mnist_noniid(x_train, y_train, num_users, num_shards, num_imgs):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset: 数据集,dataset_train
    :param num_users: 客户端数量,args.num_users
    :return:
    """
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}  # 初始化字典dict_users {0: array([], dtype=int64), 1: array([], dtype=int64), ...}
    idxs = np.arange(num_shards*num_imgs)  # [0,1,2,...,59999]
    labels = y_train  # .numpy()输出的是值，而dataset.train_labels输出的是张量  # [5 0 ... 5 6 8]  shape:(6000,)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  # 沿着第一个轴堆叠数组   # [2,60000] 第一行是index，第二行是标签
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]  # y=x.argsort() 将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    # 将标签升序排序，对应关系不变 [[0 1 2 3]            [[0 1 3 2]    第二行升序排列
    #                            [0 1 9 5]]     ->     [0 1 5 9]]
    idxs = idxs_labels[0,:]   # 排序后的index顺序

    # divide and assign
    shard = int(num_shards/num_users)
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, shard, replace=False))  # [0, 1, 2, 199]中随机选择两个组成set
        idx_shard = list(set(idx_shard) - rand_set)  # 去除随机选择的集合
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)  # concatenate进行矩阵拼接
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset: 数据集,dataset_train
    :param num_users: 客户端数量,args.num_users
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)  # 与mnist_iid一样
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_noniid(x_train, y_train, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset: 数据集,dataset_train
    :param num_users: 客户端数量,args.num_users
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}  # 初始化字典dict_users {0: array([], dtype=int64), 1: array([], dtype=int64), ...}
    idxs = np.arange(num_shards*num_imgs)  # [0,1,2,...,59999]
    labels = y_train  # .numpy()输出的是值，而dataset.train_labels输出的是张量  # [5 0 ... 5 6 8]  shape:(6000,)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  # 沿着第一个轴堆叠数组   # [2,60000] 第一行是index，第二行是标签
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]  # y=x.argsort() 将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    # 将标签升序排序，对应关系不变 [[0 1 2 3]            [[0 1 3 2]    第二行升序排列
    #                            [0 1 9 5]]     ->     [0 1 5 9]]
    idxs = idxs_labels[0,:]   # 排序后的index顺序

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))  # [0, 1, 2, 199]中随机选择两个组成set
        idx_shard = list(set(idx_shard) - rand_set)  # 去除随机选择的集合
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)  # concatenate进行矩阵拼接
    return dict_users

def fmnist_iid(x_train, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:  数据集,dataset_train
    :param num_users:  客户端数量,args.num_users
    :return: dict of image index 实例：{0: {0, 9}, 1: {5, 7}, 2: {2, 4}, 3: {3, 6}, 4: {8, 1}}  五个客户端,每个客户端有两个数据集的index
    """
    num_items = int(len(x_train)/num_users)  # 每个客户端的数据量
    dict_users, all_idxs = {}, [i for i in range(len(x_train))]
    for i in range(num_users):
        # set()函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False)) # 从all_idxs中采样，生成数据量无重复的一个字典，作为dict_users的第i个元素
        all_idxs = list(set(all_idxs) - dict_users[i]) # 取差集，删去已经被分配好的数据，直至每个用户都被分配了等量的iid数据
    return dict_users


def fmnist_noniid(x_train, y_train, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset: 数据集,dataset_train
    :param num_users: 客户端数量,args.num_users
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}  # 初始化字典dict_users {0: array([], dtype=int64), 1: array([], dtype=int64), ...}
    idxs = np.arange(num_shards*num_imgs)  # [0,1,2,...,59999]
    labels = y_train  # .numpy()输出的是值，而dataset.train_labels输出的是张量  # [5 0 ... 5 6 8]  shape:(6000,)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  # 沿着第一个轴堆叠数组   # [2,60000] 第一行是index，第二行是标签
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]  # y=x.argsort() 将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    # 将标签升序排序，对应关系不变 [[0 1 2 3]            [[0 1 3 2]    第二行升序排列
    #                            [0 1 9 5]]     ->     [0 1 5 9]]
    idxs = idxs_labels[0,:]   # 排序后的index顺序

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))  # [0, 1, 2, 199]中随机选择两个组成set
        idx_shard = list(set(idx_shard) - rand_set)  # 去除随机选择的集合
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)  # concatenate进行矩阵拼接
    return dict_users

def data_allocation(dict_users, x_train, y_train):
    data_map = []
    for i in dict_users:
        data_map[i] = {i: dict_users[i]}


