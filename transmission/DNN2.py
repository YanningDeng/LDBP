import tensorflow as tf
import os
import numpy as np

from transmission.dataProcessing import readSourceDataFromVPI, readDataFromMat, hardDecision, prepareData

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#=================必要参数设置===========================#
# 训练的总轮数
trainEpochs = 2000
# 每一批训练的数据大小
batchSize = 128
# 信息显示的频数
displayStep = 10
#读取vpi信道输出数据的文件名称
data_path='../data/-1dBm_20_100/'
datafile='result.txt'
bitsX='bits_X'
bitsY='bits_Y'
lossfile='../training_model/loss_'+datafile+"_"+str(trainEpochs)+".txt"
outputfile='../training_model/output_'+datafile+"_"+str(trainEpochs)+".txt"
#模型的存储路径
model_path = "../training_model/model_"+datafile+"_"+str(trainEpochs)+".ckpt"

#将结果写入文件
f = open(lossfile, 'w')
recordfile=open(outputfile,'w')

#测试集的比例
percent=0.3

#===========================数据准备=============================#
X_train,X_test,Y_train,Y_test=prepareData(data_path,percent,datafile)

#=============================网络参数设置============================#
# 网格参数设置--三层网络结构
n_input = 128  #
n_hidden_1 = n_input  # 第一个隐层
n_hidden_2 = n_input  # 第二个隐层
n_hidden_3 = n_input  # 第三个隐层
n_classes = n_input  # MNIST 总共10个手写数字类别
#placeholder
X = tf.placeholder(tf.float32,(n_input,2))
Y = tf.placeholder(tf.float32,(n_input,2))
# 权重参数,这里的维度都是计算好的
weights = {
    'h1': tf.Variable(tf.random_normal([n_input,n_input])),
    'h2': tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h3': tf.Variable(tf.random_normal([n_input,n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_input,n_hidden_3]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_input,2])),
    'b2': tf.Variable(tf.random_normal([n_hidden_1,2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_2,2])),
    'out': tf.Variable(tf.random_normal([n_hidden_3,2]))
}

#构建网络计算图
def multilayerPerceptron(x, weights, biases):
    """
    # 前向传播 y = wx + b
    :param x: x
    :param weights: w
    :param biases: b
    :return:
    """
    # 计算第一个隐层，使用激活函数
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(weights['h1'],tf.square(x)), biases['b1']))
    # 计算第二个隐层，使用激活函数
    layer2 = tf.nn.sigmoid(tf.add(tf.add(tf.matmul(weights['h2'],x),layer1), biases['b2']))
    # 计算第三个隐层，使用激活函数
    layer3 = tf.nn.sigmoid(tf.add(tf.matmul(weights['h3'],layer2), biases['b3']))

    # #对网络的最后一层，将结果映射到标准星座点上。

    # 计算输出层。
    outLayer = tf.add(tf.matmul(weights['out'],layer3), biases['out'])

    return outLayer

#获取预测值的score
predictValue = multilayerPerceptron(X, weights, biases)

#计算损失函数并初始化optimizer：
learnRate = 0.01
loss = tf.reduce_mean((tf.pow(predictValue-Y, 2))/(2*n_input))
optimizer = tf.train.AdamOptimizer(learning_rate=learnRate).minimize(loss)

# 验证数据
# correct_prediction = tf.equal(tf.argmax(predictValue, 1), tf.argmax(Y, 1))
# outputPoint=[]
# for sig in predictValue:    #硬判
#     outputPoint.append(hardDecision(sig))
# afterDecision=tf.convert_to_tensor(outputPoint)
# correct_prediction=tf.equal(predictValue,Y)
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

print("FUNCTIONS READY!!")



#初始化变量：
init = tf.global_variables_initializer()
#保存模型以及其所有的变量
saver=tf.train.Saver()

# print(len(train_x),len(train_allafter_x))
# print(train_x)
# print(train_allafter_x)
with tf.Session() as sess:
    # 初始化变量
    sess.run(init)
    # 训练
    for epoch in range(trainEpochs):
        avg_loss = 0.
        totalBatch = int(len(X_train)/batchSize)
        # data=tf.data.Dataset.from_tensor_slices((train_allafter_x, train_x))
        # data=data.batch(batch_size=batchSize)
        # iter_data = iter(data)
        # 遍历所有batch
        for i in range(totalBatch):
            # batchX,batchY=next(iter_data)
            batchX=X_train[i*batchSize:(i+1)*batchSize]
            batchY=Y_train[i*batchSize:(i+1)*batchSize]
            # 使用optimizer进行优化
            _, loss_value = sess.run([optimizer, loss], feed_dict={X: batchX, Y: batchY})
            # print(loss_value)
            # 求平均损失值
            avg_loss += loss_value/totalBatch
        f.write(str(loss_value)+'\n')


        # 显示信息
        if (epoch+1) % displayStep == 0:
            print("Epoch: %04d %04d Loss：%.9f" % (epoch, trainEpochs, avg_loss))
            recordfile.write("Epoch: %04d %04d Loss：%.9f" % (epoch, trainEpochs, avg_loss)+'\n')

            # train_acc = sess.run(accuracy, feed_dict={X: batchX, Y: batchY})
            # recordfile.write("Train Accuracy: %.3f" % train_acc+'\n')
            # print("Train Accuracy: %.3f" % train_acc)
            # test_acc = sess.run(accuracy, feed_dict={X: test_x[:batchSize], Y: test_allafter_x[:batchSize]})
            # recordfile.write("Test Accuracy: %.3f" % test_acc+'\n')
            # print("Test Accuracy: %.3f" % test_acc)

        res_testY = sess.run(predictValue,feed_dict={X:X_test[:batchSize]})
        recordfile.write(res_testY+'\n')

    print("Optimization Finished")
    # Save model weights to disk
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)

    #测试


f.close()
recordfile.close()