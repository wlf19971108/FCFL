import torch
import numpy as np
from torchvision import datasets
import matplotlib.pyplot as plt


def split_noniid(train_labels, alpha, n_clients):
    '''
    参数为alpha的Dirichlet分布将数据索引划分为n_clients个子集
    '''
    n_classes = train_labels.max()+1  # 共有多少类
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少   每一行是一类

    class_idcs = [np.argwhere(train_labels==y).flatten() 
           for y in range(n_classes)]
    # 记录每个K个类别对应的样本下标

    client_idcs = [[] for _ in range(n_clients)]
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    return {i: set(client_idcs[i]) for i in range(n_clients)}
    # return client_idcs

if __name__ == "__main__":

    N_CLIENTS = 10
    DIRICHLET_ALPHA = 1.0

    train_data = datasets.MNIST('./data/mnist/', train=True, download=True)
    test_data = datasets.MNIST('./data/mnist/', download=True, train=False)
    n_channels = 1


    input_size, num_class = train_data.data[0].shape[0],  len(train_data.classes)


    train_labels = np.array(train_data.targets)

    # 我们让每个client不同label的样本数量不同，以此做到Non-IID划分
    client_idcs = split_noniid(train_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)

    print(client_idcs)
    a = np.array(client_idcs)
    # new_array=[]

    print(a.shape)
    print(a)
    # for i in range(N_CLIENTS):
    #     new_array.append(len(a[i]))
    # print(new_array)
    # new_array=list(new_array)


    # # 展示不同client的不同label的数据分布
    # plt.figure(figsize=(20,3))
    # plt.hist([train_labels[idc]for idc in client_idcs], stacked=True, 
    #         bins=np.arange(min(train_labels)-0.5, max(train_labels) + 1.5, 1),
    #         label=["Client {}".format(i) for i in range(N_CLIENTS)], rwidth=0.5)
    # plt.xticks(np.arange(num_class), train_data.classes)
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(20,3))
    # plt.hist(new_array, stacked=True, 
    #         bins=100,
    #          rwidth=0.5)
    # plt.xticks(np.arange(num_class), train_data.classes)
    # plt.legend()
    # plt.show()