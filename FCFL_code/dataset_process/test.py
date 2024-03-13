#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable



def test_everyone(net_glob, x_train, y_train, x_test, y_test, args):
    with open('./save/Result.txt', 'a') as f:
        f.truncate(0)
    for idx in range(args.num_users):
        acc_train, loss_train = test_new(net_glob, x_train[idx], y_train[idx], args=args)
        acc_test, loss_test = test_new(net_glob, x_test[idx], y_test[idx], args=args)
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

def qfedavg_loss(net_g, x_test, y_test, args):
    criterion = nn.CrossEntropyLoss().to(args.device)
    net_g.eval()
    test_loss = 0
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
            batch_loss = criterion(log_probs, target.long())
            test_loss += batch_loss.item()
            # test_loss += F.cross_entropy(log_probs, target.long()).item()
    # test_loss /= num_samples
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \n'.format(test_loss))
    return test_loss

def train_performance(net_g, x_train, y_train, args):
    net_g.eval()
    train_loss = 0
    correct = 0
    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)
    num_samples = len(x_train)
    num_batchs = int(num_samples/args.bs_test)
    for k in range(num_batchs):
        start,end = k*args.bs_train,(k+1)*args.bs_train
        with torch.no_grad():
            data,target = Variable(x_train[start:end]), Variable(y_train[start:end])
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = net_g(data)
            train_loss += F.cross_entropy(log_probs, target.long()).item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]   # 最大值的位置信息
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()  # 相同的数量总计
    train_loss /= num_samples
    accuracy = 100.00 * correct / len(x_train)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            train_loss, correct, len(x_train), accuracy))
    return accuracy, train_loss


def test_performance(net_g, x_test, y_test, args):
    net_g.eval()
    test_loss = 0
    correct = 0
    x_test = torch.tensor(x_test)
    y_test = torch.tensor(y_test)
    num_samples = len(x_test)
    num_batchs = int(num_samples/args.bs_test)
    for k in range(num_batchs):
        start,end = k*args.bs_test,(k+1)*args.bs_test
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

def get_local_performance(net_g, x_test, y_test, args):
    net_g.eval()
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
            y_pred = log_probs.data.max(1, keepdim=True)[1]   # 最大值的位置信息
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()  # 相同的数量总计
    accuracy = 100.00 * correct / len(x_test)
    if args.verbose:
        print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
            correct, len(x_test), accuracy))
    return accuracy



