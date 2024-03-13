import numpy as np
from torchvision import datasets, transforms
import sys
sys.path.append('./')
from utils.options import args_parser
from client_split_sample.Dirichlet_split_datasets import split_noniid
import torch

# idxs_users = np.random.choice(range(args.num_users), m, replace=False)
# 针对上面一句修改，完成按照数据集数量完成客户端选取
# 输入：args.num_users -> int常数 所有客户端的数量 m -> int 常数 选取客户端的数量
# 输出：列表，由选中客户端的id组成  [13 45 12 16 34 20 14 17 18 42 26 43 30 46 29]

def is_all_zero(lst):
    return all(element == 0 for element in lst)

def zero_num(lst):
    num = 0
    for i in lst:
        if i == 0:
            num += 1
    return num

def non_zero_index(lst):
    index = []
    for i in range(len(lst)):
        if lst[i] != 0:
            index.append(i)
    return index

def sample_by_Q_top(num_clients_per_round, Q_value):
    if (is_all_zero(Q_value)):
        print("all zero, random select!")
        result = np.random.choice(range(len(Q_value)), num_clients_per_round, replace=False)
    elif(len(Q_value) - zero_num(Q_value) < num_clients_per_round):
        print("top + random!")
        sorted_indices = [i[0] for i in sorted(enumerate(Q_value), reverse=True, key=lambda x: x[-1])]
        result = [sorted_indices[i] for i in range(len(Q_value) - zero_num(Q_value))]
        index = non_zero_index(Q_value)
        all = [i for i in range(len(Q_value))]
        select_lst = list(set(all) - set(index))
        result_add = np.random.choice(select_lst, num_clients_per_round - (len(Q_value) - zero_num(Q_value)), replace=False)
        for item in result_add:
            result.append(item)
    else:
        # 对列表进行排序
        sorted_numbers = sorted(Q_value, reverse=True)
        # print(sorted_numbers)
        # 获取排序后的索引值
        sorted_indices = [i[0] for i in sorted(enumerate(Q_value), reverse=True, key=lambda x: x[-1])]
        result = [sorted_indices[i] for i in range(num_clients_per_round)]
    return result


def sample_by_Q_prob(num_clients_per_round, Q_value):
    if (is_all_zero(Q_value)):
        result = np.random.choice(range(len(Q_value)), num_clients_per_round, replace=False)
    else:
        sum_Q_value = np.sum(Q_value)
        probability = [Q/sum_Q_value for Q in Q_value]
        result = np.random.choice(range(len(Q_value)), num_clients_per_round, replace=False, p = probability)
    return result


if __name__ == '__main__':
#     # parse args  # python自带的命令行参数解析包，读取命令行参数
    args = args_parser()   # 读取options.py中的参数信息
    args.device = torch.device(
        f'cuda:{args.gpu}'
        if torch.cuda.is_available() and args.gpu != -1
        else 'cpu'
    )

#     torch.manual_seed(66)

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)  #训练集
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)  #测试集
        # sample users
        labels = np.array(dataset_train.targets)# 采用迪利克雷分布完成non-iid分配      
        dict_users = split_noniid(labels, alpha = 1.0, n_clients = args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)  #训练集
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)  #测试集
        labels = np.array(dataset_train.targets)# 采用迪利克雷分布完成non-iid分配      
        dict_users = split_noniid(labels, alpha = 1.0, n_clients = args.num_users)  # 为用户分配iid数据
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape  # 图像的size
    local_data_volume = [len(dict_users[cid]) for cid in range(args.num_users)]
    print(local_data_volume)
