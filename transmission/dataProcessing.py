#!/usr/bin/env python
import math

import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import scipy.io as scio

def readPredictionInTxt(filename):
    """
    Function:
        the file organize in one row,delimiter is space,element like [[float]]
    Return:
        data:np array
    """
    data=[]
    file=open(filename,'r')
    lines=file.readlines()
    strings=lines[0].strip().split(" ")
    for s in strings:
        tmp=s[2:-2]
        data.append(float(tmp))
    return np.array(data)

def hardDecision(point):
    """
    Args:
        point:[realpart,imagpart],dtype=float
    Returns:
        standard point:np.array()
    """
    real_standard_part=[-3,-1,1,3]
    minDistance=10000000000.00
    minp=0
    minq=0
    for p in real_standard_part:
        for q in real_standard_part:
            distance=(point[0]-p)**2+(point[1]-q)**2
            if(distance<minDistance):
                minp=p
                minq=q
                minDistance=distance

    # print(point,[minp,minq])
    return [minp,minq]



def hardDecisionSingalPart(points):
    """
    Args:
        points:np array,N*1
    Return:
        res:standard point,np array,N*1
    """
    res=[]
    standard=np.array([-3,-1,1,3])
    for point in points:
        distances=np.abs(point-standard)
        # print(distances)
        tmp=standard[np.argmin(distances)]
        # print(point,tmp)
        res.append(tmp)
    return np.array(res)



def readXYInTxt(filename):
    """
    Function:
        the file organize in two row,the first row is x,the second row is y,the delimiter is \t
    Args:
        filename:the name of file
    Return:
        X and Y,both np.array,dtype=float
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


def calBerFromRealImagFile(predRealFile, predImagFile, refRealFile, refImagFile,isY):
    """
    Args:
        predRealFile,predImagFile:element like [[1.0330029]],use readPredictionInTxt process
        isY:bool，是否是Y路信号

        refRealFile,refImagFile: element like a\t,use readXYInTxt process
    """
    pred_real=readPredictionInTxt(predRealFile)
    pred_imag=readPredictionInTxt(predImagFile)
    if(isY):
        _,ref_real=readXYInTxt(refRealFile)
        _,ref_imag=readXYInTxt(refImagFile)
    else:
        ref_real,_ = readXYInTxt(refRealFile)
        ref_imag,_ = readXYInTxt(refImagFile)


    bits_mps={"-3,-3":[0,0,1,0],"-3,-1":[0,0,1,1],"-3,1":[0,0,0,1],"-3,3":[0,0,0,0],
              "-1,-3":[0,1,1,0],"-1,-1":[0,1,1,1],"-1,1":[0,1,0,1],"-1,3":[0,1,0,0],
              "1,-3":[1,1,1,0],"1,-1":[1,1,1,1],"1,1":[1,1,0,1],"1,3":[1,1,0,0],
              "3,-3":[1,0,1,0],"3,-1":[1,0,1,1],"3,1":[1,0,0,1],"3,3":[1,0,0,0]}

    #组合数据
    pred_data=[]
    pred_bits=[]
    ref_data=[]
    ref_bits=[]
    for i in range(len(ref_real)):
        pred_t=[pred_real[i],pred_imag[i]]
        ref_t=[int(ref_real[i]),int(ref_imag[i])]

        pred_t=hardDecision(pred_t) #硬判决


        #转换成字符串以在字典中匹配
        predStr=",".join(str(v) for v in pred_t)
        refStr=",".join(str(v) for v in ref_t)
        pred_data.append(pred_t)
        ref_data.append(ref_t)
        pred_bits.extend(bits_mps[predStr])
        ref_bits.extend(bits_mps[refStr])
    print(pred_bits)
    print(ref_bits)
    pred_bits_np=np.array(pred_bits)
    ref_bits_np=np.array(ref_bits)

    return sum(pred_bits_np!=ref_bits_np)/len(pred_bits)

def prepareData(data_path, test_percent, inFile,outFile,isY):
    """
    Function:prepare data for network training
    Args:
        data_path
        test_percent: test data / total data
        inFile:the filename of vpi input data
        outFile:the filename of vpi output data
        isY:bool,是否是Y路信号
    Return:
        train_X,test_X,train_Y,test_Y:np.array
    """
    alldata_x,alldata_y = readXYInTxt(data_path+inFile)

    afterVPI_x, afterVPI_y = readXYInTxt(data_path + outFile)

    #随机抽取
    if(isY):
        X_train,X_test,Y_train,Y_test=train_test_split(afterVPI_y,alldata_y,test_size=test_percent,random_state=0)
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(afterVPI_x, alldata_x, test_size=test_percent,
                                                            random_state=0)

    return X_train,X_test,Y_train,Y_test



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

def readTestInTxt(filename):
    data = []
    file = open(filename, 'r')
    lines = file.readlines()
    strings = lines[0].strip().split(" ")
    for s in strings:

        data.append(int(float(s)))
    print(data)
    return np.array(data)
#
# X_train,X_test,Y_train,Y_test=prepareData("../data/-1dBm_20_100/",0.4,"-1dBm-20x100.mat")
# print(X_train)
# print(X_test)
# print(Y_train)
# print(Y_test)

# readXYAfterDSPInTxt("result_x.txt")

# X_train, X_test, Y_train, Y_test = prepareData("../data/-1dBm_20_100/", 0.3,"ref.txt", "result.txt")
# print(len(X_train))
# print(X_train)
# print(Y_train)


def calAllDataBerY():
    """
    计算所有功率下的数据的预测ber
    isY：bool，是否是Y路信号
    """
    dataDir="/Users/deng/MySources/Optics/Code/ldbp_dyn/LDBP/data"
    allDirectorys=os.listdir(dataDir)
    berFile=open(dataDir+"/ber_y.txt","w")

    print(allDirectorys)
    for dir in allDirectorys:
        if os.path.isdir(dataDir+"/"+dir):
            predRealFile=dataDir+"/"+dir+"/prediction/real_y_"+dir+"_2000.txt"
            predImagFile=dataDir+"/"+dir+"/prediction/imag_y_"+dir+"_2000.txt"
            refRealFile=dataDir+"/"+dir+"/ref_real.txt"
            refImagFile=dataDir+"/"+dir+"/ref_imag.txt"
            ber=calBerFromRealImagFile(predRealFile,predImagFile,refRealFile,refImagFile,True)
            print(dir,ber)
            berFile.write(dir+" "+str(ber)+"\n")
    berFile.close()

def calAllDataBerX():
    """
    计算X路信号所有功率下的数据的预测ber
    """
    dataDir="/Users/deng/MySources/Optics/Code/ldbp_dyn/LDBP/data"
    allDirectorys=os.listdir(dataDir)
    berFile=open(dataDir+"/ber_x.txt","w")

    print(allDirectorys)
    for dir in allDirectorys:
        if os.path.isdir(dataDir+"/"+dir):
            predRealFile=dataDir+"/"+dir+"/prediction/real_x_"+dir+"_2000.txt"
            predImagFile=dataDir+"/"+dir+"/prediction/imag_x_"+dir+"_2000.txt"
            refRealFile=dataDir+"/"+dir+"/ref_real.txt"
            refImagFile=dataDir+"/"+dir+"/ref_imag.txt"
            ber=calBerFromRealImagFile(predRealFile,predImagFile,refRealFile,refImagFile,False)
            print(dir,ber)
            berFile.write(dir+" "+str(ber)+"\n")
    berFile.close()
def calAllDataBerY():
    """
    计算X路信号所有功率下的数据的预测ber
    """
    dataDir="/Users/deng/MySources/Optics/Code/ldbp_dyn/LDBP/data"
    allDirectorys=os.listdir(dataDir)
    berFile=open(dataDir+"/ber_y.txt","w")

    print(allDirectorys)
    for dir in allDirectorys:
        if os.path.isdir(dataDir+"/"+dir):
            predRealFile=dataDir+"/"+dir+"/prediction/real_y_"+dir+"_2000.txt"
            predImagFile=dataDir+"/"+dir+"/prediction/imag_y_"+dir+"_2000.txt"
            refRealFile=dataDir+"/"+dir+"/ref_real.txt"
            refImagFile=dataDir+"/"+dir+"/ref_imag.txt"
            ber=calBerFromRealImagFile(predRealFile,predImagFile,refRealFile,refImagFile,True)
            print(dir,ber)
            berFile.write(dir+" "+str(ber)+"\n")
    berFile.close()
# calAllDataBerX()
# calAllDataBerY()

