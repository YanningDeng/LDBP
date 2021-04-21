
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
def draw_loss(filename):
    """

    """
    file=open(filename,'r')
    loss=[]
    for line in file:
        line=line.strip('\n')
        if "Loss:" in line:
            line.find(":")
            loss.append(eval(line[line.find("Loss:")+5:]))
    print(loss)
    plt.plot(np.linspace(9,1999,200),loss)
    plt.xlabel('训练迭代次数')
    plt.ylabel('loss')
    plt.show()

def draw_train_accuracy(filename):
    """

    """
    file=open(filename,'r')
    train_acc=[]
    for line in file:
        line=line.strip('\n')
        if "Train Accuracy:" in line:
            train_acc.append(eval(line[line.find(":")+1:]))
    print(train_acc)
    plt.plot(np.linspace(9,1999,200),train_acc)
    # # plt.xlabel('训练迭代次数')
    plt.ylabel('train accuracy')
    plt.show()

def draw_test_accuracy(filename):
    """

    """
    file = open(filename, 'r')
    train_acc = []
    for line in file:
        line = line.strip('\n')
        if "Test Accuracy:" in line:
            train_acc.append(eval(line[line.find(":") + 1:]))
    print(train_acc)
    plt.plot(np.linspace(9, 1999, 200), train_acc)
    # # plt.xlabel('训练迭代次数')
    plt.ylabel('test accuracy')
    plt.show()

draw_test_accuracy('record.txt')