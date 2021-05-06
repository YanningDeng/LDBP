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
本代码训练X路和Y路信号的实部，保存模型，模型在实部和虚部上进行预测
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

def loadModelAndTest(vpiDir,isY):
    """
    Args:
        isY:bool,是否是Y路信号
    """
    # 一些必要的参数
    epochs = 2000
    batchSize=128
    filenameEnd = vpiDir + "_" + str(epochs)

    # 划分训练集和测试集的输入输出
    # 读取vpi信道输出数据的文件名称
    data_path = '../data/'+vpiDir+'/'
    realInFile='ref_real.txt'
    realOutFile='result_real.txt'
    imagInFile = 'ref_imag.txt'  # 这里只需要填文件名，路径拼接在函数里实现了
    imagOutFile = 'result_imag.txt' #这个文件是Matlab处理后的数据

    # 模型的存储路径
    model_path = "../training_model/"+vpiDir+"/model_x" + filenameEnd
    X_imag,_ = readXYInTxt(data_path + imagOutFile)
    Y_imag,_ = readXYInTxt(data_path + imagInFile)
    X_real,_ = readXYInTxt(data_path + realOutFile)
    Y_real,_ = readXYInTxt(data_path + realInFile)
    if(isY):
        model_path = "../training_model/" + vpiDir + "/model_y" + filenameEnd
        _,X_imag=readXYInTxt(data_path+imagOutFile)
        _,Y_imag=readXYInTxt(data_path+imagInFile)
        _,X_real=readXYInTxt(data_path+realOutFile)
        _,Y_real=readXYInTxt(data_path+realInFile)


    # 加载已保存的模型
    model=make_model(batchSize)
    model.load_weights(model_path, by_name=False)

    # 预测并评估结果
    pred_imag = model.predict(X_imag)
    pred_imag_filename=data_path + "prediction/imag_x_" + filenameEnd + ".txt"
    if(isY):
        pred_imag_filename=data_path + "prediction/imag_y_" + filenameEnd + ".txt"
    print(pred_imag_filename)
    file = open(pred_imag_filename, "wt")
    for data in pred_imag:
        file.write(str(data) + " ")
    file.close()

    pred_real=model.predict(X_real)
    pred_real_filename = data_path + "prediction/real_x_" + filenameEnd + ".txt"
    if(isY):
        pred_real_filename = data_path + "prediction/real_y_" + filenameEnd + ".txt"
    print(pred_real_filename)
    file = open(pred_real_filename, "wt")
    for data in pred_real:
        file.write(str(data) + " ")
    file.close()

    standardPoints_imag = hardDecisionSingalPart(pred_imag)
    standardPoints_real=hardDecisionSingalPart(pred_real)
    print(np.count_nonzero(standardPoints_imag == Y_imag) / len(Y_imag))
    print(np.count_nonzero(standardPoints_real == Y_real) / len(Y_real))


# if __name__ == '__main__':
def trainAndSaveModel(vpiDir,isY):
    #========================一些必要的参数===========================#
    epochs=2000
    batchSize=128 #模型输入数据的size
    filenameEnd=vpiDir+"_"+str(epochs)

    #划分训练集和测试集的输入输出
    # 读取vpi信道输出数据的文件名称
    data_path = '../data/'+vpiDir+'/'
    inFile = 'ref_real.txt' #这里只需要填文件名，路径拼接在函数里实现了
    outFile='result_real.txt'
    # 模型的存储路径
    model_path = "../training_model/"+vpiDir+"/model_x" + filenameEnd
    if(isY):
        model_path = "../training_model/"+vpiDir+"/model_y" + filenameEnd
    # 测试集的比例
    percent = 0.3

    #=================================================================#

    # ===========================数据准备=============================#
    X_train, X_test, Y_train, Y_test = prepareData(data_path, percent, inFile,outFile,isY)
    model=make_model(batchSize)
    # total_real_X = np.append(X_train, X_test)
    #================================================================#

    #训练模型并保存为model

    train_history=model.fit(X_train,Y_train,batch_size=batchSize,epochs=epochs,verbose=1,validation_data=(X_test,Y_test))
    model.save_weights(model_path)

    #加载已保存的模型
    model.load_weights(model_path,by_name=False)

    #预测并评估实部结果

    # pred=model.predict(total_real_X)

    # file=open(data_path+"pred_real"+filenameEnd+".txt", "wt")
    # for data in pred:
    #     file.write(str(data)+" ")
    # file.close()


    epochs = range(len(train_history.history['loss']))
    plt.figure()
    plt.plot(epochs, train_history.history['loss'], 'b', label='Training loss')
    plt.plot(epochs, train_history.history['val_loss'], 'r', label='Validation loss')
    plt.title('Loss')
    plt.legend()


    if(isY):
        plt.savefig(data_path+'loss_real_y'+filenameEnd+".jpg")
    else:
        plt.savefig(data_path + 'loss_real_x' + filenameEnd + ".jpg")


    #计算验证准确率
    # standardPoints = hardDecisionSingalPoint(points)
    # print(np.count_nonzero(standardPoints == ref_test) / len(ref_test))

def userSpecificModelTestData(modelDir,vpiDir):
    """
    Function:用制定的的模型测试其他功率的数据
    """
    epochs = 2000
    batchSize = 128
    filenameEnd = vpiDir + "_" + str(epochs)

    data_path = '../data/' + vpiDir + '/'
    inFile = 'ref_imag.txt'  # 这里只需要填文件名，路径拼接在函数里实现了
    outFile = 'result_imag.txt'  # 这个文件是Matlab处理后的数据
    # 模型的存储路径
    model_path = "../training_model/"+vpiDir+"/model" + filenameEnd
    X_imag, _ = readXYInTxt(data_path + outFile)
    Y_imag, _ = readXYInTxt(data_path + inFile)
    # 加载已保存的模型
    model = make_model(batchSize)
    model.load_weights(model_path, by_name=False)

    # 预测并评估结果
    pred_imag = model.predict(X_imag)
    standardPoints = hardDecisionSingalPart(pred_imag)
    print(np.count_nonzero(standardPoints == Y_imag) / len(Y_imag))


for i in range(-4,5):
    start_time=time.time()
    vpiDir=str(i)+"dBm_20_100"
    # powerString=str(i)+"dBm"
    # 训练并测试Y路信号
    trainAndSaveModel(vpiDir,True)
    loadModelAndTest(vpiDir,True)
    # 训练并测试X路信号
    trainAndSaveModel(vpiDir, False)
    loadModelAndTest(vpiDir,False)
    end_time=time.time()
    print("time:",end_time-start_time)

# modelDir="-2dBm_20_100"
# userSpecificModelTestData(modelDir,vpiDir)



