import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:  数据集,dataset_train
    :param num_users:  客户端数量,args.num_users
    :return: dict of image index 实例：{0: {0, 9}, 1: {5, 7}, 2: {2, 4}, 3: {3, 6}, 4: {8, 1}}  五个客户端,每个客户端有两个数据集的index
    """
    num_items = int(len(dataset)/num_users)  # 每个客户端的数据量
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        # set()函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False)) # 从all_idxs中采样，生成数据量无重复的一个字典，作为dict_users的第i个元素
        all_idxs = list(set(all_idxs) - dict_users[i]) # 取差集，删去已经被分配好的数据，直至每个用户都被分配了等量的iid数据
    return dict_users


def mnist_noniid(dataset, num_users):
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
    labels = dataset.train_labels.numpy()  # .numpy()输出的是值，而dataset.train_labels输出的是张量  # [5 0 ... 5 6 8]  shape:(6000,)

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


if __name__ == '__main__':
    print("ok")
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
