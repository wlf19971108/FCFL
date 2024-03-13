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

def sample_by_proportion(dic_client, num_clients_per_round, F_T):
    # sourcery skip: assign-if-exp, inline-immediately-returned-variable
    args = args_parser()
    local_data_volume = [len(dic_client[cid]) for cid in range(args.num_users)]
    total_data_volume = sum(local_data_volume)
    probability = [l_d_v / total_data_volume for l_d_v in local_data_volume]
    if args.num_users > 0: 
        selected_clients = np.random.choice(args.num_users, num_clients_per_round, replace=F_T, p = probability)
    else:
        selected_clients = []
    # print(probability)
    return selected_clients


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
