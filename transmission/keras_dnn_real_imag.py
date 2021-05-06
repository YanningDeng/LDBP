import time

import keras
import pandas as pd
from sklearn.datasets import load_boston
import tensorflow as tf
from keras.layers import Dense,Dropout,Activation,Input
from keras.models import *

from keras import optimizers
import numpy as np
from numpy import *
from keras import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_absolute_error

"""
本代码训练X路信号的实部和虚部并分别保存模型，然后在Y路信号上预测。
"""


#神经网络模型构建
from dataProcessing import prepareData

from transmission.dataProcessing import readXYInTxt, readPredictionInTxt, readTestInTxt, hardDecisionSingalPart


def make_model(InputSize):
    model=Sequential()
    model.add(Dense(units=128,activation='relu',input_shape=(InputSize,1)))
    model.add(Dropout(0.05))
    model.add(Dense(units=64,activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(units=32,activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(units=1, activation=None))
    sgd=optimizers.SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
    model.compile(loss='mean_squared_error',optimizer=sgd,metrics=[metrics.mae])
    # print(model.summary())
    return model

def loadModelAndTest(vpiDir,isPredictY,isImag,isTraningY):
    """
    Functions:根据是否是实部或者虚部的模型，预测X路以及Y路信号
    Args:
        isPredictY:bool,是否预测的是Y路信号
        isTraningY:bool,是否训练的是Y路信号的模型
        isImag:bool,是否是实部信号
    """
    # 一些必要的参数
    epochs = 2000
    batchSize=128
    filenameEnd = vpiDir + "_" + str(epochs)

    # 划分训练集和测试集的输入输出
    # 读取vpi信道输出数据的文件名称
    data_path = '../data/'+vpiDir+'/'
    dataInFile = 'ref_real.txt'
    dataOutFile = 'result_real.txt'
    if(isImag):
        dataInFile='ref_imag.txt'# 这里只需要填文件名，路径拼接在函数里实现了
        dataOutFile='result_imag.txt'#这个文件是Matlab处理后的数据

    # 读取测试数据
    X, _ = readXYInTxt(data_path + dataOutFile)
    Y, _ = readXYInTxt(data_path + dataInFile)
    if (isPredictY):
        _, X = readXYInTxt(data_path + dataOutFile)
        _, Y = readXYInTxt(data_path + dataInFile)


    # 模型的存储路径,根据X路和Y路以及实部虚部来加载不同的模型
    if (isTraningY):
        if (isImag):
            model_path = "../training_model/" + vpiDir + "/imag_model_y" + filenameEnd
        else:
            model_path = "../training_model/" + vpiDir + "/real_model_y" + filenameEnd
    else:
        if (isImag):
            model_path = "../training_model/" + vpiDir + "/imag_model_x" + filenameEnd
        else:
            model_path = "../training_model/" + vpiDir + "/real_model_x" + filenameEnd


    # 加载已保存的模型
    model=make_model(batchSize)
    model.load_weights(model_path, by_name=False)

    # 预测并评估结果
    pred = model.predict(X)
    if(isImag):
        if(isPredictY):
            pred_filename=data_path + "prediction/imag_y_" + filenameEnd + ".txt"
        else:
            pred_filename = data_path + "prediction/imag_x_" + filenameEnd + ".txt"
    else:
        if (isPredictY):
            pred_filename = data_path + "prediction/real_y_" + filenameEnd + ".txt"
        else:
            pred_filename = data_path + "prediction/real_x_" + filenameEnd + ".txt"
    print(pred_filename)
    file = open(pred_filename, "wt")
    for data in pred:
        file.write(str(data) + " ")
    file.close()

    standardPoints = hardDecisionSingalPart(pred)
    print(np.count_nonzero(standardPoints == Y) / len(Y))

# if __name__ == '__main__':
def trainAndSaveModel(vpiDir,isY,isImag):
    """
    Functions:训练X路或者Y路信号中的实部或者虚部模型并保存
    """
    #========================一些必要的参数===========================#
    epochs=2000
    batchSize=128 #模型输入数据的size
    filenameEnd=vpiDir+"_"+str(epochs)

    #划分训练集和测试集的输入输出
    # 读取vpi信道输出数据的文件名称
    data_path = '../data/'+vpiDir+'/'
    inFile = 'ref_real.txt' #这里只需要填文件名，路径拼接在函数里实现了
    outFile='result_real.txt'
    if(isImag):
        inFile = 'ref_imag.txt'  # 这里只需要填文件名，路径拼接在函数里实现了
        outFile = 'result_imag.txt'
    # 模型的存储路径
    if(isY):
        if(isImag):
            model_path = "../training_model/" + vpiDir + "/imag_model_y" + filenameEnd
        else:
            model_path = "../training_model/"+vpiDir+"/real_model_y" + filenameEnd
    else:
        if (isImag):
            model_path = "../training_model/" + vpiDir + "/imag_model_x" + filenameEnd
        else:
            model_path = "../training_model/" + vpiDir + "/real_model_x" + filenameEnd
    # 测试集的比例
    percent = 0.3

    #=================================================================#

    # ===========================数据准备=============================#
    X_train, X_test, Y_train, Y_test = prepareData(data_path, percent, inFile,outFile,False)
    model=make_model(batchSize)
    # total_real_X = np.append(X_train, X_test)
    #================================================================#

    #训练模型并保存为model

    train_history=model.fit(X_train,Y_train,batch_size=batchSize,epochs=epochs,verbose=1,validation_data=(X_test,Y_test))
    model.save_weights(model_path)

    #加载已保存的模型
    model.load_weights(model_path,by_name=False)


    epochs = range(len(train_history.history['loss']))
    plt.figure()
    plt.plot(epochs, train_history.history['loss'], 'b', label='Training loss')
    plt.plot(epochs, train_history.history['val_loss'], 'r', label='Validation loss')
    plt.title('Loss')
    plt.legend()

    #根据X路和Y路以及实部和虚部命名loss图像
    if(isY):
        if(isImag):
            plt.savefig(data_path+'loss_imag_y'+filenameEnd+".jpg")
        else:
            plt.savefig(data_path + 'loss_real_y' + filenameEnd + ".jpg")
    else:
        if (isImag):
            plt.savefig(data_path + 'loss_imag_x' + filenameEnd + ".jpg")
        else:
            plt.savefig(data_path + 'loss_real_x' + filenameEnd + ".jpg")


for i in range(-4,5):
    start_time=time.time()
    vpiDir=str(i)+"dBm_20_100"
    # 训练X路信号的实部和虚部
    trainAndSaveModel(vpiDir,False,True)
    trainAndSaveModel(vpiDir, False, False)
    #在Y路信号的实部和虚部上测试
    loadModelAndTest(vpiDir,True,True,False)
    loadModelAndTest(vpiDir,True,False,False)
    #在X路信号的实部和虚部上测试
    loadModelAndTest(vpiDir, False, True, False)
    loadModelAndTest(vpiDir, False, False, False)
    end_time=time.time()
    print("time:",end_time-start_time)



