import keras
import pandas as pd
from sklearn.datasets import load_boston
import tensorflow as tf
from keras.layers import Dense,Dropout,Activation,Input
from keras.models import Sequential,Model
from keras import optimizers
import numpy as np
from numpy import *
from keras import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_absolute_error

#神经网络模型构建
from dataProcessing import prepareData


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
    print(model.summary())
    return model

if __name__ == '__main__':

    #一些必要的参数
    epochs=2000
    batchSize=128 #模型输入数据的size

    #划分训练集和测试集的输入输出
    # 读取vpi信道输出数据的文件名称
    data_path = '../data/-1dBm_20_100/'
    inFile = 'ref.txt' #这里只需要填文件名，路径拼接在函数里实现了
    outFile='result.txt'
    # 模型的存储路径
    model_path = "../training_model/model_" + str(epochs) + ".ckpt"
    # 测试集的比例
    percent = 0.3

    # ===========================数据准备=============================#
    X_train, X_test, Y_train, Y_test = prepareData(data_path, percent, inFile,outFile)
    print("------------")
    print(X_train.dtype)
    print(len(X_train),len(X_test))
    model=make_model(batchSize)

    #训练模型并保存为module.h5

    model.fit(X_train,Y_train,batch_size=batchSize,epochs=epochs,verbose=1,validation_data=(X_test,Y_test))
    model.save_weights(model_path)

    #加载已保存的模型
    model.load_weights(model_path,by_name=False)

    #预测并评估结果
    pred=model.predict(X_test)
    file=open("pred.txt","wt")
    file.write(pred)
    file.close()

    plt.plot(Y_test,label='True')
    plt.plot(pred,label='NN')
    plt.legend()
    plt.show()
    score=r2_score(Y_test,pred)
    error=mean_absolute_error(Y_test,pred)
    print(score)
    print(error)