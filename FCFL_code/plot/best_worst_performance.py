import pylab
import matplotlib.pyplot as plt
import numpy as np


#从.txt中读取数据
def loadData(fileName):
    inFile = open(fileName, 'r') # 以只读方式打开某filename文件
    #定义2个空的list，用来存放文件中的数据
    x = []
    y = [] # training acc
    z = [] # test acc
    for line in inFile:
        trainingSet = line.split(':')#对于每一行，按','把数据分开，这里是分成两部分
        # print(trainingSet)
        x.append(trainingSet[0])#第一部分，即文件中的第一列数据逐一添加到list x中
        special_chars = " Testing accuracy"
        for char in special_chars:
            trainingSet[2] = trainingSet[2].replace(char, "")
        y.append(float(trainingSet[2]))#第二部分，即文件中的第二列数据逐一添加到list y中
        z.append(float(trainingSet[3]))
    return x, y, z

def worst_fraction(acc_list, fraction):
    acc_sort = sorted(acc_list, reverse=False)
    worst = [acc_sort[i] for i in range(int(len(acc_list)*fraction))]
    return sum(worst)/len(worst)

def best_fraction(acc_list, fraction):
    acc_sort = sorted(acc_list, reverse=True)
    best = [acc_sort[i] for i in range(int(len(acc_list)*fraction))]
    return sum(best)/len(best)

if __name__ == '__main__':
    x, y, z = loadData('./save/Result.txt')
    # y为training acc z为test acc
    print(worst_fraction(y, 0.1))
    print(best_fraction(y, 0.1))
    print(f"training acc Variance: {np.var(y)}")
    print(f"test acc Variance: {np.var(z)}")
