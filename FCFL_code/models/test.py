#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs_test)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

def test_img_everyone(net_g, datatest, idxs, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(DatasetSplit(datatest, idxs), batch_size=args.local_bs, shuffle=True)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


def test_everyone(net_glob, dataset_train, dataset_test, dict_users_train, dict_users_test, args):
    for idx in range(args.num_users):
        acc_train, loss_train = test_img_everyone(net_glob, dataset_train, idxs=dict_users_train[idx], args=args)
        acc_test, loss_test = test_img_everyone(net_glob, dataset_test, idxs=dict_users_test[idx], args=args)
        # print(f"Client:{idx}")
        # print("Training accuracy: {:.2f}".format(acc_train))
        # print("Testing accuracy: {:.2f}".format(acc_test))
        with open('./save/Result.txt', 'a') as f:
            f.write(str(idx) + ': Training accuracy: ' + str(float(acc_train)) + ' Testing accuracy: ' + str(float(acc_test)) + '\n')


def test_new(net_g, x_test, y_test, args):
    net_g.eval()
    test_loss = 0
    correct = 0
    x_test = torch.tensor(x_test)
    y_test = torch.tensor(y_test)
    num_samples = len(x_test)
    num_batchs = int(num_samples/args.bs_test)
    for k in range(num_batchs):
        start,end = k*args.bs_train,(k+1)*args.bs_train
        with torch.no_grad():
            data,target = Variable(x_test[start:end]), Variable(y_test[start:end])
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = net_g(data)
            test_loss += F.cross_entropy(log_probs, target.long()).item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]   # 最大值的位置信息
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()  # 相同的数量总计
    test_loss /= num_samples
    accuracy = 100.00 * correct / len(x_test)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(x_test), accuracy))
    return accuracy, test_loss



