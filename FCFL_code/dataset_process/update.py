import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from torch.autograd import Variable
import torch.nn.functional as F
import copy
    

def split_cifar(x_train, y_train, idxs):
    image = []
    label = []
    for i in idxs:
        image.append(x_train[i])
        label.append(y_train[i])
    image = torch.tensor([item.cpu().detach().numpy() for item in image]).cuda()
    image = image.reshape(len(image),x_train.shape[1],x_train.shape[2],x_train.shape[3])
    label = torch.tensor(label)
    # print(image.shape)   # torch.Size([600, 1, 28, 28])
    # print(label.shape)   # torch.Size([600])
    return image, label

def split_mnist(x_train, y_train, idxs):
    image = []
    label = []
    for i in idxs:
        image.append(x_train[i])
        label.append(y_train[i])
    image = torch.tensor(image)
    image = image.reshape(len(image),x_train.shape[1],x_train.shape[2],x_train.shape[3])
    label = torch.tensor(label)
    # print(image.shape)   # torch.Size([600, 1, 28, 28])
    # print(label.shape)   # torch.Size([600])
    return image, label


def fcfl_local_train_old(net_glob, x_train, y_train, args):  
    net_glob.train()
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    loss_func = nn.CrossEntropyLoss()
    epoch_loss = []

    for iter in range(args.local_ep):
        batch_loss = []
        num_samples = len(x_train)
        # num_batchs = int(num_samples/args.bs_train)
        num_batchs = max(int(num_samples/args.bs_train), 1)
        for k in range(num_batchs):
            start,end = k*args.bs_train, (k+1)*args.bs_train
            data,target = Variable(x_train[start:end],requires_grad=False).to(args.device), Variable(y_train[start:end]).to(args.device)
            net_glob.zero_grad()
            log_probs = net_glob(data)
            loss = loss_func(log_probs, target.long())
            loss.backward()
            optimizer.step()
            if args.verbose and k % 10 == 0:
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(iter, k * len(data), len(x_train),100. * k / len(x_train), loss.item()))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))

    net_glob.eval()
    acc_local = []
    for k in range(num_batchs):
        start,end = k*args.bs_train, (k+1)*args.bs_train
        data,target = Variable(x_train[start:end],requires_grad=True).to(args.device), Variable(y_train[start:end]).to(args.device)
        net_glob.zero_grad()
        log_probs = net_glob(data)
        _, predicted = torch.max(log_probs, dim=1)
        acc = torch.sum(predicted == target.long()).item()/len(predicted)
        acc_local.append(acc)
    accuracy = sum(acc_local)/len(acc_local)
    
    return net_glob.state_dict(), sum(epoch_loss) / len(epoch_loss), accuracy

def fcfl_local_train(net_glob, x_train, y_train, x_test, y_test, args):  
    net_glob.train()
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    loss_func = nn.CrossEntropyLoss()
    epoch_loss = []

    for iter in range(args.local_ep):
        batch_loss = []
        num_samples_train = len(x_train)
        # num_batchs = int(num_samples/args.bs_train)
        num_batchs_train = max(int(num_samples_train/args.bs_train), 1)
        for k in range(num_batchs_train):
            start,end = k*args.bs_train, (k+1)*args.bs_train
            data,target = Variable(x_train[start:end],requires_grad=False).to(args.device), Variable(y_train[start:end]).to(args.device)
            net_glob.zero_grad()
            log_probs = net_glob(data)
            loss = loss_func(log_probs, target.long())
            loss.backward()
            optimizer.step()
            if args.verbose and k % 10 == 0:
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(iter, k * len(data), len(x_train),100. * k / len(x_train), loss.item()))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    
    net_glob.eval()
    acc_local_train = []
    for k in range(num_batchs_train):
        start_train, end_train = k*args.bs_train, (k+1)*args.bs_train
        # print(start_train)
        # print(end_train)
        data_train, target_train = Variable(x_train[start_train:end_train],requires_grad=True).to(args.device), Variable(y_train[start_train:end_train]).to(args.device)
        # print(data_train)
        net_glob.zero_grad()
        log_probs_train = net_glob(data_train)
        # print(log_probs_train)
        _, predicted_train = torch.max(log_probs_train, dim=1)
        # print(len(predicted_train))
        # print(torch.sum(predicted_train == target_train.long()).item())
        acc_train = torch.sum(predicted_train == target_train.long()).item()/len(predicted_train)
        acc_local_train.append(acc_train)
    accuracy_train = sum(acc_local_train)/len(acc_local_train)

    num_samples_test = len(x_test)
    # num_batchs = int(num_samples/args.bs_train)
    num_batchs_test = max(int(num_samples_test/args.bs_test), 1)
    acc_local_test = []
    for k in range(num_batchs_test):
        start_test,end_test = k*args.bs_test, (k+1)*args.bs_test
        data_test,target_test = Variable(x_test[start_test:end_test],requires_grad=True).to(args.device), Variable(y_test[start_test:end_test]).to(args.device)
        net_glob.zero_grad()
        log_probs_test = net_glob(data_test)
        _, predicted_test = torch.max(log_probs_test, dim=1)
        acc_test= torch.sum(predicted_test == target_test.long()).item()/len(predicted_test)
        acc_local_test.append(acc_test)
    accuracy_test = sum(acc_local_test)/len(acc_local_test)

    accuracy = (accuracy_train + accuracy_test) / 2

    return net_glob.state_dict(), sum(epoch_loss) / len(epoch_loss), accuracy


def local_train(net_glob, x_train, y_train, args):  
    net_glob.train()
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    loss_func = nn.CrossEntropyLoss()
    epoch_loss = []
    epoch_acc = []

    for iter in range(args.local_ep):
        batch_loss = []
        batch_acc =[]
        num_samples = len(x_train)
        num_batchs = max(int(num_samples/args.bs_train), 1)
        # print(num_batchs)

        for k in range(num_batchs):
            start,end = k*args.bs_train, (k+1)*args.bs_train
            data,target = Variable(x_train[start:end],requires_grad=False).to(args.device), Variable(y_train[start:end]).to(args.device)
            net_glob.zero_grad()
            log_probs = net_glob(data)
            loss = loss_func(log_probs, target.long())
            # loss = F.cross_entropy(log_probs, target.long())
            _, predicted = torch.max(log_probs, dim=1)
            acc = torch.sum(predicted == target.long()).item()/len(predicted)
            loss.backward()
            optimizer.step()
            if args.verbose and k % 10 == 0:
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(iter, k * len(data), len(x_train),100. * k / len(x_train), loss.item()))
            batch_loss.append(loss.item())
            batch_acc.append(acc)
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        epoch_acc.append(sum(batch_acc)/len(batch_acc))

    acc_local = []
    for k in range(num_batchs):
        start,end = k*args.bs_train, (k+1)*args.bs_train
        data,target = Variable(x_train[start:end],requires_grad=True).to(args.device), Variable(y_train[start:end]).to(args.device)
        
        net_glob.zero_grad()
        log_probs = net_glob(data)
        _, predicted = torch.max(log_probs, dim=1)
        acc = torch.sum(predicted == target.long()).item()/len(predicted)
        acc_local.append(acc)
    acc_test = sum(acc_local)/len(acc_local)

        # Compute loss on the whole training data
    # comp_loss = []
    # for k in range(num_batchs):
    #     start,end = k*args.bs_train, (k+1)*args.bs_train
    #     data,target = Variable(x_train[start:end],requires_grad=True).to(args.device), Variable(y_train[start:end]).to(args.device)
    #     # print(data.shape)
    #     # print(target)
    #     # images, labels = data.to(args.device), target.to(args.device)
    #     net_glob.zero_grad()
    #     log_probs = net_glob(data)
    #     loss = loss_func(log_probs, target.long())
    #     comp_loss.append(loss.detach().item())
    # comp_loss = sum(comp_loss)/len(comp_loss)

    return net_glob.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_test


def fedavg_local_train(net_glob, x_train, y_train, args):  
    net_glob.train()
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    loss_func = nn.CrossEntropyLoss()
    epoch_loss = []
    for iter in range(args.local_ep):
        batch_loss = []
        num_samples = len(x_train)
        num_batchs = max(int(num_samples/args.bs_train), 1)

        for k in range(num_batchs):
            start,end = k*args.bs_train, (k+1)*args.bs_train
            data,target = Variable(x_train[start:end],requires_grad=False).to(args.device), Variable(y_train[start:end]).to(args.device)
            net_glob.zero_grad()
            log_probs = net_glob(data)
            loss = loss_func(log_probs, target.long())
            # print(loss)
            loss.backward()
            optimizer.step()
            if args.verbose and k % 10 == 0:
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(iter, k * len(data), len(x_train),100. * k / len(x_train), loss.item()))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    return net_glob.state_dict(), sum(epoch_loss) / len(epoch_loss)

def qfedavg_local_train(net_glob, x_train, y_train, args):  
    old_weights = copy.deepcopy(net_glob.state_dict())
    net_glob.train()
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    loss_func = nn.CrossEntropyLoss()
    epoch_loss = []

    for iter in range(args.local_ep):
        batch_loss = []
        num_samples = len(x_train)
        num_batchs = int(num_samples/args.bs_train)

        for k in range(num_batchs):
            start,end = k*args.bs_train, (k+1)*args.bs_train
            data,target = Variable(x_train[start:end],requires_grad=True).to(args.device), Variable(y_train[start:end]).to(args.device)
            # print(data.shape)
            # print(target)
            # images, labels = data.to(args.device), target.to(args.device)
            net_glob.zero_grad()
            log_probs = net_glob(data)
            loss = loss_func(log_probs, target.long())
            # loss = F.cross_entropy(log_probs, target.long())
            loss.backward()
            optimizer.step()
            if args.verbose and k % 10 == 0:
                print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(iter, k * len(data), len(x_train),100. * k / len(x_train), loss.item()))
            batch_loss.append(loss.detach().item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    difference = copy.deepcopy(old_weights)
    with torch.no_grad():
        for key in difference.keys():
            difference[key] = net_glob.state_dict()[key] - old_weights[key]
    return difference, sum(epoch_loss) / len(epoch_loss)


