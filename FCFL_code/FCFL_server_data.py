import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from datetime import datetime
from time import strftime
import collections

# sourcery skip: dont-import-test-modules
from dataset_process.load_dataset import test_train_allocation, test_train_mixture, server_client_split, test_train_split, mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, fmnist_iid, fmnist_noniid  # 引入了三种iid与non-iid数据库类型
from utils.options import args_parser
from dataset_process.load_dataset import load_mnist, load_cifar, load_fmnist, load_synthetic_train, load_synthetic_test
from client_split_sample.Dirichlet_split_datasets import split_noniid
from client_selection.sampling_by_proportion import sample_by_proportion
from client_selection.sampling_by_Q import sample_by_Q_top, is_all_zero
from dataset_process.update import fcfl_local_train, split
from utils.weight_cal import para_diff_cal, float_mulpty_OrderedDict, weight_sum, orderdict_zeros_like
from models.Nets import MLP, CNNMnist, CNNCifar, CNNFashionMnist, MCLR_Logistic
from models.aggregation import weight_agg, acc_global_estimate
from dataset_process.test import test_new, test_everyone


"""


"""


if __name__ == '__main__':
    args = args_parser()   # 读取options.py中的参数信息
    args.device = torch.device(f'cuda:{args.gpu}'if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        x_train, y_train = load_mnist('./dataset/mnist', kind = 'train')
        x_train = x_train.reshape(x_train.shape[0],1,28,28)
        x_test, y_test = load_mnist('./dataset/mnist',kind='t10k')
        x_test = x_test.reshape(x_test.shape[0],1,28,28)

        x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
        x_train, x_test = x_train/255, x_test/255

        """
        正则化
        """
        std = 0.3081

        mean_train, mean_test = np.random.uniform(0.1307,0.1307,size = (60000, 1, 28, 28)), np.random.uniform(0.1307,0.1307,size = (10000, 1, 28, 28))
        x_train, x_test = (x_train - mean_train)/std, (x_test - mean_test)/std
        x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)

        """
        分发数据
        """
        x_all, y_all = test_train_mixture(x_train, y_train, x_test, y_test)
        # print(x_all.shape)
        # print(y_all.shape)
        img_size = x_all[0].shape  # [1, 28, 28]
        img_size = torch.tensor(img_size)

        ratio_client_server = 0.8 # 服务器客户端数据分配
        server_data, server_label, client_data, client_label = server_client_split(x_all, y_all, ratio_client_server)
        

        ratio_train_test = 0.8  # 客户端本地划分训练集测试集，训练集占比      
        if args.iid:
            dict_users = mnist_iid(client_data, args.num_users)  # 为用户分配iid数据
        else:
            # labels = np.array(client_label)# 采用迪利克雷分布完成non-iid分配      
            # np.random.seed(args.seed)
            # dict_users = split_noniid(labels, alpha = args.dirichlet, n_clients = args.num_users)
            dict_users = mnist_noniid(client_data, client_label, args.num_users, 200, 280)
        dict_users_train, dict_users_test = test_train_split(dict_users, ratio_train_test)
    elif args.dataset == 'cifar':
        # trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)  #训练集
        # dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)  #测试集
        x_train, y_train, x_test, y_test = load_cifar('./dataset/cifar/cifar-10-batches-py')
        x_train, x_test = x_train.reshape(x_train.shape[0], 3, 32, 32), x_test.reshape(x_test.shape[0], 3, 32, 32)
        x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
        x_train, x_test = x_train/255, x_test/255

        """
        正则化
        """
        channel_mean, channel_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        mean_train_0 = np.random.uniform(channel_mean[0],channel_mean[0],size = (50000, 1, 32, 32))
        mean_train_1 = np.random.uniform(channel_mean[1],channel_mean[1],size = (50000, 1, 32, 32))
        mean_train_2 = np.random.uniform(channel_mean[2],channel_mean[2],size = (50000, 1, 32, 32))
        mean_train = np.concatenate((mean_train_0, mean_train_1, mean_train_2), axis = 1)
        mean_test_0 = np.random.uniform(channel_mean[0],channel_mean[0],size = (10000, 1, 32, 32))
        mean_test_1 = np.random.uniform(channel_mean[1],channel_mean[1],size = (10000, 1, 32, 32))
        mean_test_2 = np.random.uniform(channel_mean[2],channel_mean[2],size = (10000, 1, 32, 32))
        mean_test = np.concatenate((mean_test_0, mean_test_1, mean_test_2), axis = 1)
        for i in range(3):
            x_train[:,i,:,:] = (x_train[:,i,:,:] - mean_train[:,i,:,:]) / channel_std[i]
            x_test[:,i,:,:] = (x_test[:,i,:,:] - mean_test[:,i,:,:]) / channel_std[i]

        x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
        x_train, x_test = torch.tensor(x_train), torch.tensor(x_test)

        x_all, y_all = test_train_mixture(x_train, y_train, x_test, y_test)
        img_size = x_all[0].shape  # [3, 32, 32]
        img_size = torch.tensor(img_size)
        ratio_train_test = 0.8

        if args.iid:
            dict_users = cifar_iid(x_all, args.num_users)  # 为用户分配iid数据
            dict_users_train, dict_users_test = test_train_split(dict_users, ratio_train_test)
        else:
            labels = np.array(y_all)# 采用迪利克雷分布完成non-iid分配      
            np.random.seed(args.seed)
            dict_users = split_noniid(labels, alpha = args.dirichlet, n_clients = args.num_users)
            # dict_users = mnist_noniid(x_all, y_all, args.num_users)  # 否则为用户分配non-iid数据
        dict_users_train, dict_users_test = test_train_split(dict_users, ratio_train_test)
    elif args.dataset == 'fmnist':
        x_train, y_train = load_fmnist('./dataset/fmnist', kind='train')
        x_test, y_test = load_fmnist('./dataset/fmnist', kind='t10k')
        x_train, x_test = x_train.reshape(x_train.shape[0],1,28,28), x_test.reshape(x_test.shape[0],1,28,28)
        x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
        x_train, x_test = x_train/255, x_test/255

        """
        正则化
        """
        mean, std = 0.2860, 0.3530
        mean_train = np.random.uniform(mean,mean,size = (60000, 1, 28, 28))
        mean_test = np.random.uniform(mean,mean,size = (10000, 1, 28, 28))

        x_train, x_test = (x_train - mean_train)/std, (x_test - mean_test)/std
        x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)

        """
        分发数据
        """
        x_train_new, x_test_new, y_train_new, y_test_new = test_train_allocation(x_train, y_train, x_test, y_test, 1/7)
        img_size = x_train_new[0].shape  # [1, 28, 28]
        img_size = torch.tensor(img_size)

        if args.iid:
            dict_users = fmnist_iid(x_train_new, args.num_users)  # 为用户分配iid数据
            dict_users_test = fmnist_iid(x_test_new, args.num_users)
        else:
            dict_users = fmnist_noniid(x_train_new, y_train_new, args.num_users)  # 否则为用户分配non-iid数据
            labels_train = np.array(y_test_new)# 采用迪利克雷分布完成non-iid分配      
            np.random.seed(args.seed)
            dict_users_test = split_noniid(labels_train, alpha = args.dirichlet, n_clients = args.num_users)
    elif args.dataset == 'synthetic':
        print("synthetic dataset")
    else:
        exit('Error: unrecognized dataset')

    if args.dataset == 'synthetic':
        image, label = load_synthetic_train()
        image_test, label_test = load_synthetic_test()
    else:
        """
        根据dict_users将数据集进行分发
        """
        # image[i]存储着客户端i的数据，label[i]存储着客户端i的标签
        # image和label是字典   键是客户端i
        image = {}
        label = {}
        image_test = {}
        label_test = {}
        for i in dict_users:  
            image[i], label[i] = split(client_data, client_label, dict_users_train[i])
            image_test[i], label_test[i] = split(client_data, client_label, dict_users_test[i])
        # print(image[0].shape)  # torch.Size([560, 1, 28, 28])
        # print(label[0].shape)  # torch.Size([560])
        # for i in range(args.num_users):
        #     print(image[i].shape)

        """
        # x_train_performance, x_test_performance, y_train_performance, y_test_performance用于测试
        # print(x_train_performance.shape)  # (56000, 1, 28, 28)
        # print(x_test_performance.shape)   # (14000, 1, 28, 28)
        # print(y_train_performance.shape)   # (56000,)
        # print(y_test_performance.shape)    # (14000,)
        """
    x_train_performance = copy.deepcopy(image[0])
    x_test_performance = copy.deepcopy(image_test[0])
    y_train_performance = copy.deepcopy(label[0])
    y_test_performance = copy.deepcopy(label_test[0])
    for i in range(1, args.num_users):
        x_train_performance = np.concatenate([x_train_performance,image[i]], axis=0)
        x_test_performance = np.concatenate([x_test_performance,image_test[i]], axis=0)
        y_train_performance = np.concatenate([y_train_performance,label[i]], axis=0)
        y_test_performance = np.concatenate([y_test_performance,label_test[i]], axis=0)

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'fmnist':
        net_glob = CNNFashionMnist(args=args).to(args.device)
    elif args.model == 'mclr' and args.dataset == 'synthetic':
        net_glob = MCLR_Logistic(60, 10).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    # FL主体部分
    loss_train_plot_list = []
    acc_train_plot_list = []
    # select_num = [0] * args.num_users
    # local_data = [len(dict_users[cid]) for cid in range(args.num_users)]
    # print(local_data)
    Acc_local = np.zeros((args.num_users, args.epochs), float)
    select_history = []

    for iter in range(args.epochs):
        w_glob = net_glob.state_dict()
        loss_locals = []  # 对于每一个epoch，初始化worker的损失
        w_locals = []  # 存储客户端本地权重
        acc_locals = []  # 存储客户端本地准确率

        Acc_local_list = []  # 所有客户端的本地性能
        for i in range(args.num_users):
            net_glob.eval()
            # 同时使用训练集与测试集
            # acc_train, _ = test_new(net_glob, image[i], label[i], args=args)
            # acc_test, _ = test_new(net_glob, image_test[i], label_test[i], args=args)
            # acc_train, acc_test = float(acc_train), float(acc_test)
            # local_acc = (acc_train + acc_test)/200
            # Acc_local_list.append(local_acc)
            # 仅使用测试集
            acc_test, _ = test_new(net_glob, image_test[i], label_test[i], args=args)
            acc_test = float(acc_test)
            local_acc = acc_test/100
            Acc_local_list.append(local_acc)
            Acc_local[i][iter] = local_acc
        print("performance list:")
        print(Acc_local[:,iter])
        print("var:")
        print(np.var(Acc_local[:,iter]*100))
        if iter>0:
            print("performance increase:")
            print(np.mean(Acc_local[:,iter])-np.mean(Acc_local[:,iter-1]))
        if iter>1:
            print("selected client performance increase:")
            for i in select_history[iter-1]:
                print(Acc_local[i,iter-1]-Acc_local[i,iter-2])

        net_glob.train()
        if iter == 0:
            # 初始化不公平累计队列Q
            Q = np.zeros((args.num_users, args.epochs), float)
            for i in range(args.num_users):
                Q[i][0] = 0.0
            # print(Q)
            # 选取客户端
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            select_history.append(idxs_users)
            # 聚合权重的选择：根据数据量；1/m
            local_data_volume = [len(dict_users[cid]) for cid in idxs_users]
            # print(local_data_volume)
            total_data_volume = sum(local_data_volume)
            # print(total_data_volume)
            weights = [l_d_v / total_data_volume for l_d_v in local_data_volume]   # 权重与数据量成正比
            # weights = [1/m] * m
            # print(weights)
        else:
            # 计算uf
            uf = [0.0] * args.num_users
            for i in range(args.num_users):
                uf[i] = Acc_global - Acc_local_list[i] if Acc_global > Acc_local_list[i] else 0
            print("uf:")
            print(uf)
            # print(min(uf))
            # 更新Q
            alpha = 5.0
            Q_value = []
            weights_all_clients = [0.0] * args.num_users
            # print(weights)
            for i, j in zip(idxs_users, range(m)):
                weights_all_clients[i] = weights[j]
            # print(weights_all_clients)
            # print(weights)

            for i in range(args.num_users):
                Q[i][iter] = max(Q[i][iter-1] + alpha * uf[i] - weights_all_clients[i], 0)
                Q_value.append(Q[i][iter])
            print(Q_value)
            # 选择客户端
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = sample_by_Q_top(m, Q_value)  # 根据Q选取top-m个客户端
            select_history.append(idxs_users)
            print(idxs_users)
            sum_Q = sum(Q[i][iter] for i in idxs_users)
            # print(weights)
            weights = []
            # print(weights)
            if sum_Q == 0 and is_all_zero(Q[:, iter]):
                # 聚合权重的选择：根据数据量；1/m
                local_data_volume = [len(dict_users[cid]) for cid in idxs_users]
                # print(local_data_volume)
                total_data_volume = sum(local_data_volume)
                # print(total_data_volume)
                weights = [l_d_v / total_data_volume for l_d_v in local_data_volume]
                # weights = [1/m] * m
                # print(weights)
            else:
                for i in idxs_users:
                    weight_clients = Q[i][iter]/sum_Q
                    weights.append(weight_clients)
                    # print(weights)
        # print("Q:")
        # print(Q[:, iter])
        # print(idxs_users)
        # print(weights)

        # for i in idxs_users:
        #     select_num[i] = select_num[i] + 1
        
        # 本地训练
        for idx in idxs_users:  # 对于选取的m个worker
            w, loss, acc_client = fcfl_local_train(copy.deepcopy(net_glob).to(args.device), image[idx], label[idx], image_test[idx], label_test[idx], args)
            
            w_locals.append(w)  
            loss_locals.append(loss)
            acc_locals.append(acc_client)
        # 全局模型聚合
        wnew = weight_agg(w_locals, weights)
        # 全局性能评估
        net_glob.eval()
        Acc_global, _ = test_new(net_glob, server_data, server_label, args)
        Acc_global = float(Acc_global/100)
        # 全局性能估计
        # Acc_global = acc_global_estimate(acc_locals, weights)
        # Acc_global = float(Acc_global)
        
        print("estimation of global performance:")
        print(Acc_global)
        net_glob.load_state_dict(wnew)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train_plot_list.append(loss_avg)
        # # print acc
        net_glob.eval()
        acc_train_plot, _ = test_new(net_glob, x_train_performance, y_train_performance, args)
        print('Round {:3d}, Average acc {:.3f}'.format(iter, acc_train_plot))
        acc_train_plot_list.append(acc_train_plot)

    # print(select_num)
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train_plot_list)), loss_train_plot_list)
    plt.ylabel('train_loss')
    plt.savefig('./save/fcfl_loss_{}_{}_{}_alpha{}_C{}_iid{}_{}.png'.format(args.dataset, args.model, args.epochs, alpha, args.frac, args.iid, datetime.now().strftime("%H_%M_%S")))
    # plot acc curve
    plt.figure()
    plt.plot(range(len(acc_train_plot_list)), acc_train_plot_list)
    plt.ylabel('train_acc')
    plt.savefig('./save/fcfl_acc_{}_{}_{}_alpha{}_C{}_iid{}_{}.png'.format(args.dataset, args.model, args.epochs, alpha, args.frac, args.iid, datetime.now().strftime("%H_%M_%S")))


    # testing  测试集上进行测试
    net_glob.eval()
    acc_train, loss_train = test_new(net_glob, x_train_performance, y_train_performance, args)
    acc_test, loss_test = test_new(net_glob, x_test_performance, y_test_performance, args)

    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    # print("#######################################################")
    """
    test_everyone函数
    # for idx in range(args.num_users):
    #     acc_train, loss_train = test_new(net_glob, x_train[idx], y_train[idx], args=args)
    #     acc_test, loss_test = test_new(net_glob, x_test[idx], y_test[idx], args=args)
    """
    test_everyone(net_glob, image, label, image_test, label_test, args)


