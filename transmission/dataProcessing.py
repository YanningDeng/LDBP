#!/usr/bin/env python
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import scipy.io as scio

def readXYInTxt(filename):
    """
    Function:
        the file organize in two row,the first row is x,the second row is y,the delimiter is \t
    Args:
        filename:the name of file
    Return:
        X and Y,both np.array
    """
    file=open(filename,'r')
    lines=file.readlines()
    lineX=lines[0].strip().split("\t")
    lineY=lines[1].strip().split("\t")

    X=[]
    Y=[]

    for strx in lineX:
        X.append(float(strx))
    for stry in lineY:
        Y.append(float(stry))
    return np.array(X),np.array(Y)

def prepareData(data_path, test_percent, inFile,outFile):
    """
    Function:prepare data for network training
    Args:
        data_path
        test_percent: test data / total data
        inFile:the filename of vpi input data
        outFile:the filename of vpi output data
    Return:
        train_X,test_X,train_Y,test_Y
    """
    alldata_x,alldata_y = readXYInTxt(data_path+inFile)

    # alldata_y = readSourceDataFromVPI(data_path + "bits_Y")

    # X_Y=np.vstack((alldata_x,alldata_y))

    afterVPI_x, afterVPI_y = readXYInTxt(data_path + outFile)

    # afterVPI_X_Y=np.vstack((afterVPI_x,afterVPI_y))
    #随机抽取
    X_train,X_test,Y_train,Y_test=train_test_split(afterVPI_x,alldata_x,test_size=test_percent,random_state=0)

    return X_train,X_test,Y_train,Y_test

def hardDecision(sig):
    """
    Function: Return the most closest standard point
    Args:
        sig: len is 2.
    Return
        standard signal
    """
    standard_point=[[-3,-3],[-3,-1],[-3,1],[-3,3],[-1,-3],[-1,-1],[-1,1],[-1,3],[1,-3],[1,-1],[1,1],[1,3],[3,-3],[3,-1],[3,1],[3,3]]
    mindiff=10000000000000
    res=[]
    for s in standard_point:
        diff=np.power(s[0]-sig[0],2)+np.power(s[1]-sig[1],2)
        if diff<mindiff:
            mindiff=diff
            res=s
    return res

def next_batch(batchSize,data,index):
    """
    Args:
        batchSize:the size of batch,if data length is not enough,

    """

def getTrainTestDataFromDir(dirname,percent):
    """
    Args:
        dirname:it contains some sub directory,each directory have three file,bits_X,bits_Y,0dBm_15_80.mat。
        percent: traindata/Total
    Return:
        train_src_cpx_x,train_src_cpy_y,test_src_cpx_x,test_src_cpy_y,train_ch_cpx_x,train_ch_cpy_y,test_ch_cpx_x,test_ch_cpy_y
    """
    dirs = os.listdir(dirname)
    train_src_cpx_x=[]
    train_src_cpy_y=[]
    test_src_cpx_x = []
    test_src_cpy_y = []
    train_ch_cpx_x = []
    train_ch_cpy_y = []
    test_ch_cpx_x = []
    test_ch_cpy_y = []

    # 输出所有文件和文件夹
    for file in dirs:
        # 得到该文件下所有目录的路径
        m = os.path.join(dirname, file)
        # 判断该路径下是否是文件夹
        if (os.path.isdir(m)):
            for datafile in os.listdir(m):
                if datafile=='bits_X':
                    tmp_x=readSourceDataFromVPI(os.path.join(m,datafile))
                    train_src_cpx_x.extend(tmp_x[:len(tmp_x)*percent])
                    test_src_cpx_x.extend(tmp_x[len(tmp_x)*percent:])
                elif datafile=='bits_Y':
                    tmp_y = readSourceDataFromVPI(os.path.join(m, datafile))
                    train_src_cpy_y.extend(tmp_y[:len(tmp_y) * percent])
                    test_src_cpy_y.extend(tmp_y[len(tmp_y) * percent:])
                else:
                    tmp_x,tmp_y=readDataFromMat(os.path.join(m,datafile))
                    train_ch_cpx_x.extend(tmp_x[:len(tmp_x)*percent])
                    train_ch_cpy_y.extend(tmp_y[:len(tmp_y)*percent])
                    test_ch_cpx_x.extend(tmp_x[len(tmp_x):])
                    test_ch_cpy_y.extend(tmp_y[len(tmp_y)])

    return np.asarray(train_src_cpx_x),np.asarray(test_src_cpx_x),\
            np.asarray(train_src_cpy_y),np.asarray(test_src_cpy_y),\
            np.asarray(train_ch_cpx_x),np.asarray(test_ch_cpx_x),\
            np.asarray(train_ch_cpy_y),np.asarray(test_ch_cpy_y)

def readDataFromMat(filename):
    """
    Args:
        filename:file contains data output from VPI,usually named as kdBm-5*80.mat
    Returns:
        x,y:[[Ix,Qx]],[[Iy,Qy]]
    """
    data = scio.loadmat(filename)
    Ix=data['Ix_sig'][0]
    Iy=data['Iy_sig'][0]
    Qx=data['Qx_sig'][0]
    Qy=data['Qy_sig'][0]

    print("Ix length:")
    print(len(Ix))

    x=np.empty([len(Ix), 2], dtype=float)
    y=np.empty([len(Ix), 2], dtype=float)

    for i in range(len(Ix)):
        x[i]=[Ix[i],Qx[i]]
        y[i]=[Iy[i],Qy[i]]
    # complex_x = 1j * Qx+Ix
    # complex_y = 1j * Qy+Iy
    # return np.asarray(complex_x),np.asarray(complex_y)
    return x,y

def convertNDarryToImage(nums,filename):
    """
    Args:
        nums: numpy array
    Functions:
        convert numpy array to tiff format , float type array can save preserve lossless
        save picture as format tiff ,named filename.
    """
    img = Image.fromarray(nums).convert('tiff')
    img.save(filename)


def readComplexDataAfterDSPInTxt(filename):
    """
    Args:
        filename:the file contains complex data after dsp
    Returns:
        data:numpy array
    """
    file = open(filename, 'r')
    data = []
    for line in file:
        s=eval(line.strip('\n').replace('i','j'))
        data.append(complex(s))
        # print(data)
    # print(data)
    file.close()
    return np.array(data)


def readSourceDataFromVPI(filename):
    """
    Args:
        filname:source data filename from vpi output,named bits_X or bits_Y,
    Return:
        data:numpy array,[[realx,imgx]]
    """
    grapCode={'0000':-3+3j,'0100':-1+3j,'1100':1+3j,'1000':3+3j,'0001':-3+1j,'0101':-1+1j,'1101':1+1j,'1001':3+1j,'0011':-3-1j,'0111':-1-1j,'1111':1-1j,'1011':3-1j,'0010':-3-3j,'0110':-1-3j,'1110':1-3j,'1010':3-3j}
    file = open(filename)
    # 忽略前5行的无效输出
    for i in range(5):
        file.readline()

    line = file.readline()  # read total signal num
    line = line.strip('\n')
    line=line.replace(' ', '')
    totalSignalNum = len(line)//4
    # print(totalSignalNum)

    data = np.empty([totalSignalNum, 2], dtype=int)
    data_complex=np.empty([totalSignalNum, 1],dtype=complex)  #处理了复数形式但是没有用到
    for i in range(totalSignalNum):
        curSig=line[4*i:4*i+4]
        data_complex[i]=grapCode[curSig]
        data[i]=[float(np.real(grapCode[curSig])),float(np.imag(grapCode[curSig]))]
    file.close()
    return data
#
# X_train,X_test,Y_train,Y_test=prepareData("../data/-1dBm_20_100/",0.4,"-1dBm-20x100.mat")
# print(X_train)
# print(X_test)
# print(Y_train)
# print(Y_test)

# readXYAfterDSPInTxt("result_x.txt")

X_train, X_test, Y_train, Y_test = prepareData("../data/-1dBm_20_100/", 0.3,"ref.txt", "result.txt")
print(len(X_train))
print(X_train)
print(Y_train)
