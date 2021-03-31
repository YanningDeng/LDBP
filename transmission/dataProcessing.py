#!/usr/bin/env python
import numpy as np


def readSourceDataFromVPI(filename):
    """
    Args:
        filname:source data filename from vpi output,named bits_X or bits_Y,
    Return:
        real,img:numpy array
    """
    file = open(filename)
    # 忽略前5行的无效输出
    for i in range(5):
        line = file.readline()

    line = file.readline()  # read total signal num
    line = line.strip('\n')
    line=line.split(" ")
    print(line)
    totalSignalNum = len(line)//4
    print(totalSignalNum)

    real=np.empty(totalSignalNum,dtype=int)
    img=np.empty(totalSignalNum,dtype=int)
    for i in range(totalSignalNum):
        a =eval( line[4*i])
        b=eval(line[4*i+1])
        c=eval(line[4*i+2])
        d=eval(line[4*i+3])
        realSig=b+a*2
        imgSig=d+c*2
        real[i]=realSig
        img[i]=imgSig
        print(a,b,c,d,realSig,imgSig)

    # print(line)

    file.close()
    return real,img


realx,imgx=readSourceDataFromVPI("bits_X.txt")
realy,imgy=readSourceDataFromVPI("bits_Y.txt")
