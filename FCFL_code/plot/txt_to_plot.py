import pylab
import matplotlib.pyplot as plt


#从.txt中读取数据
def loadData(fileName):
    inFile = open(fileName, 'r') # 以只读方式打开某filename文件
    #定义2个空的list，用来存放文件中的数据
    x = []
    y = []
    for line in inFile:
        trainingSet = line.split(':')#对于每一行，按','把数据分开，这里是分成两部分
        # print(trainingSet)
        x.append(trainingSet[0])#第一部分，即文件中的第一列数据逐一添加到list x中
        special_chars = " Testing accuracy"
        for char in special_chars:
            trainingSet[2] = trainingSet[2].replace(char, "")
        y.append(float(trainingSet[2]))#第二部分，即文件中的第二列数据逐一添加到list y中

    return x, y

#绘制该文件中的数据
def plotData(x, y):
    # length = len(y)
    # pylab.figure(1)
    # pylab.plot(x, y, 'ko')#'ko'表示点的类型为黑色实心圆点
    # pylab.xlabel('Horizontal axis title')
    # pylab.ylabel('Vertical axis title')
    # pylab.show()#让绘制的图像在屏幕上显示出来
    plt.plot(x, y)
    plt.title('Histogram of accuracy of every clients')
    plt.xlabel('Client')
    plt.ylabel('Accuracy')
    plt.savefig('./save/fedavg_acc_every_client.png')
    plt.show()
    

 
if __name__ == '__main__':
    x, y = loadData('./save/Result.txt')
    plotData(x, y)
