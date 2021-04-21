import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#数据准备


#placeholder
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

#准备权重参数：
# 网格参数设置--三层网络结构
n_input = 784  # MNIST 数据输入(28*28*1=784)
n_hidden_1 = 512  # 第一个隐层
n_hidden_2 = 256  # 第二个隐层
n_hidden_3 = 128  # 第三个隐层
n_classes = 10  # MNIST 总共10个手写数字类别

# 权重参数
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
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
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # 计算第二个隐层，使用激活函数
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
    # 计算第三个隐层，使用激活函数
    layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
    # 计算第输出层。
    outLayer = tf.add(tf.matmul(layer3, weights['out']), biases['out'])

    return outLayer

#获取预测值的score
predictValue = multilayerPerceptron(X, weights, biases)

#计算损失函数并初始化optimizer：
learnRate = 0.01
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictValue, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learnRate).minimize(loss)

# 验证数据
correct_prediction = tf.equal(tf.argmax(predictValue, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
print("FUNCTIONS READY!!")



#初始化变量：
init = tf.global_variables_initializer()

#在session中执行graph定义的运算
# 训练的总轮数
trainEpochs = 20
# 每一批训练的数据大小
batchSize = 128
# 信息显示的频数
displayStep = 5


with tf.Session() as sess:
    # 初始化变量
    sess.run(init)
    # 训练
    for epoch in range(trainEpochs):
        avg_loss = 0.
        totalBatch = int(mnist.train.num_examples/batchSize)
        # 遍历所有batch
        for i in range(totalBatch):
            batchX, batchY = mnist.train.next_batch(batchSize)
            # 使用optimizer进行优化
            _, loss_value = sess.run([optimizer, loss], feed_dict={X: batchX, Y: batchY})
            # 求平均损失值
            avg_loss += loss_value/totalBatch

        # 显示信息
        if (epoch+1) % displayStep == 0:
            print("Epoch: %04d %04d Loss：%.9f" % (epoch, trainEpochs, avg_loss))
            train_acc = sess.run(accuracy, feed_dict={X: batchX, Y: batchY})
            print("Train Accuracy: %.3f" % train_acc)
            test_acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
            print("Test Accuracy: %.3f" % test_acc)

    print("Optimization Finished")