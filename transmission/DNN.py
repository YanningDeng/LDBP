import tensorflow as tf
import os
import numpy as np

from transmission.dataProcessing import readSourceDataFromVPI, readDataFromMat

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#数据准备

#训练集和测试集的比例
percent=0.7

alldata_x=readSourceDataFromVPI('../data/-1dBm_5_80/bits_X')

#复数信号求模
alldata_y=readSourceDataFromVPI('../data/-1dBm_5_80/bits_Y')

allafter_x,allafter_y=readDataFromMat('../data/-1dBm_5_80/-1dBm-5x80.mat')
train_x=alldata_x[:int(len(alldata_x)*percent)]
test_x=alldata_x[int(len(alldata_x)*percent):]
#接收端4倍采样
allafter_x=allafter_x[::4]
allafter_y=allafter_y[::4]

train_allafter_x=allafter_x[:len(train_x)]
test_allafter_x=allafter_y[len(train_x):]


#准备权重参数：
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
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(weights['h2'],layer1), biases['b2']))
    # 计算第三个隐层，使用激活函数
    layer3 = tf.nn.sigmoid(tf.add(tf.matmul(weights['h3'],layer2), biases['b3']))
    # 计算第输出层。
    outLayer = tf.add(tf.matmul(weights['out'],layer3), biases['out'])

    return outLayer

#获取预测值的score
predictValue = multilayerPerceptron(X, weights, biases)

#计算损失函数并初始化optimizer：
learnRate = 0.01
loss = tf.reduce_mean((tf.pow(predictValue-Y, 2))/(2*n_input))
optimizer = tf.train.AdamOptimizer(learning_rate=learnRate).minimize(loss)

# 验证数据
correct_prediction = tf.equal(tf.argmax(predictValue, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
print("FUNCTIONS READY!!")

#初始化变量：
init = tf.global_variables_initializer()

#在session中执行graph定义的运算
# 训练的总轮数
trainEpochs = 3000
# 每一批训练的数据大小
batchSize = 128
# 信息显示的频数
displayStep = 10
f = open("loss_3000.txt", 'w')
record=open("record_3000.txt",'w')

# print(len(train_x),len(train_allafter_x))
# print(train_x)
# print(train_allafter_x)
with tf.Session() as sess:
    # 初始化变量
    sess.run(init)
    # 训练
    for epoch in range(trainEpochs):
        avg_loss = 0.
        totalBatch = int(len(train_allafter_x)/batchSize)
        # data=tf.data.Dataset.from_tensor_slices((train_allafter_x, train_x))
        # data=data.batch(batch_size=batchSize)
        # iter_data = iter(data)
        # 遍历所有batch
        for i in range(totalBatch):
            # batchX,batchY=next(iter_data)
            batchX=train_allafter_x[i*batchSize:(i+1)*batchSize]
            batchY=train_x[i*batchSize:(i+1)*batchSize]
            # 使用optimizer进行优化
            _, loss_value = sess.run([optimizer, loss], feed_dict={X: batchX, Y: batchY})

            # print(loss_value)
            # 求平均损失值
            avg_loss += loss_value/totalBatch
        f.write(str(loss_value) + '\n')
        # 显示信息
        if (epoch+1) % displayStep == 0:
            print("Epoch: %04d %04d Loss：%.9f" % (epoch, trainEpochs, avg_loss))
            train_acc = sess.run(accuracy, feed_dict={X: batchX, Y: batchY})
            record.write("Epoch: %04d %04d Loss：%.9f\n" % (epoch, trainEpochs, avg_loss))
            print("Train Accuracy: %.3f" % train_acc)
            record.write("Train Accuracy: %.3f\n" % train_acc)
            test_acc = sess.run(accuracy, feed_dict={X: test_x[:batchSize], Y: test_allafter_x[:batchSize]})
            record.write("Test Accuracy: %.3f\n" % test_acc)
            print("Test Accuracy: %.3f" % test_acc)

    print("Optimization Finished")
f.close()
record.close()