import tensorflow as tf
import numpy as np
import os


def myLinearRegression(self, train_X, train_Y, training_epochs, learning_rate):
    """
    使用TensorFlow中的API实现的一个简单的回归模型,网络的输入的shape为(float32,dim,2)
    :return: None
    """
    data_num = train_X.shape[0]
    # n = 684  # 神经元数量
    input_X = tf.reduce_sum(tf.square(train_X
                                      ), axis=2)

    n = data_num
    with tf.variable_scope("data"):
        # 实现步骤：
        # 第一步：准备好线性回归模型的数据
        # 实现模型 y = w x +b
        X = tf.placeholder(tf.float32, shape=(None, data_num, 2), name='X')
        Y = tf.placeholder(tf.float32, shape=(
            None, data_num, 2), name='Y')  # 注意数据类型

    with tf.variable_scope("model"):
        # 第二步：构建线性回归模型
        # 1.初始化 weight 和 bias ,使用变量定义
        # 因为这两个系数需要训练 Variable 默认是可被训练的类型
        stddev = 2 / np.sqrt(n)
        weight = tf.Variable(tf.truncated_normal(
            (n, n), stddev=stddev), name="weight")

        bias = tf.Variable(tf.zeros([n]), name='bias')

    with tf.variable_scope("loss"):
        # 3.线性回归模型建立，激活函数使用relu
        y_tmp = tf.matmul(X, weight) + bias
        y_predict = tf.nn.relu(y_tmp)

        # matched filter
        y = cconv(y, ps_filter)  # complex(y) = complex(x) * real(h)
        y = tf.complex(y[:, :, 0], y[:, :, 1])
        # downsample
        y = y[:, ::OS_d] / tf.complex(tf.sqrt(P_W), 0.0) / np.sqrt(OS_d)
        # constant phase-offset rotation
        tmp = tf.reduce_sum(tf.conj(x)*y, 1, keepdims=True)
        phi_cpe = -tf.atan2(tf.imag(tmp), tf.real(tmp))
        x_hat = y * tf.exp(tf.complex(0.0, phi_cpe))

        mean_squared_error = tf.reduce_mean(tf.square(tf.abs(x-x_hat)))

        # 第三步：利用均方误差计算线性回归的损失(loss)

        loss = tf.reduce_mean(tf.square(Y-y_predict))/(2*n)

        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(loss)

    # 定义一个保存模型的实例
    saver = tf.compat.v1.train.Saver()
    # 2.初始化变量(全局变量初始化)
    variable_op = tf.compat.v1.global_variables_initializer()
    with tf.Session() as sess:
        # 运行全局变量初始化这个OP
        sess.run(variable_op)
        print("初始化时,模型的权重为{},偏值为{}".format(weight.eval(), bias.eval()))
        # 先判断模型文件是否存在
        if os.path.exists("./temp/checkpoint"):
            # 加载训练模型
            saver.restore(sess, "./temp/model")
            print("模型加载后为,模型的权重为{},偏值为{}".format(
                weight.eval()[0][0], bias.eval()))

        for epoch in range(training_epochs):
            for (x, y) in zip(train_X, train_Y):
                sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})

            print("训练{}次后,模型的权重为{},偏值为{}".format(
                epoch, weight.eval(), bias.eval()))
        # 保存训练好的模型
        saver.save(sess, "./temp/model")
        # 上面就已经结束了。计算一下loss 、W、b的值
        print("optimization Finished")
        training_cost = sess.run(loss, feed_dict={X: train_X, Y: train_Y})
        print("Training cost=", training_cost, "weight=",
              sess.run(weight), "b=", sess.run(bias), "\n")
        y_hat = sess.run(y_predict)
        # 画图
        # plt.plot(train_X, train_Y, 'ro', label="Original data")
        # plt.plot(train_X, sess.run(weight)*train_X +
        #          sess.run(bias), label="Fitted line")
        # plt.legend()
        # plt.show()

    return y_hat


def buildNetWork(self, input, output):
    """
    Input Args:
        input:输入数据的围度，单位为复数
        output:输出数据的围度，单位为复数
    Output Args:
        y:网络输出结果
        loss
    """
    n_inputs = input
    n_hidden1 = 300
    n_hidden2 = 100
    n_output = output
    x = tf.placeholder(tf.complex64, shape=(None, n_inputs), name='x')
    y = tf.placeholder(tf.complex64, shape=(
        None, n_inputs), name='y')  # 注意数据类型

    # 搭建网络
    with tf.name_scope("dnn"):
        hidden1 = neuron_layer(x, n_hidden1, "hidden1", activation="relu")
        hidden2 = neuron_layer(
            hidden1, n_hidden2, "hidden2", activation="relu")
        y = neuron_layer(hidden2, n_output, "output")

    # 这里的网络搭建也可以选择tensorflow的全连接层
    # with tf.name_scope("dnn"):
    #     hidden1 = fully_connected(X, n_hidden1, scope="hidden1")  # 激活函数默认为relu
    #     hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
    #     logits = fully_connected(
    #         hidden2, n_outputs, scope="outputs", activation_fn=None)

    # 使用reduce_mean()计算loss。
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.square(tf.abs(y-x)), name="loss")
    return
# 构建隐藏层


def neuron_layer(X, n_neurons, name, activation=None):
    """
        Args:
            X:输入
            n_neurons:
        Return:
            y:Wx*b
    """
    with tf.name_scope(name):  # 为了方便在TensorBoard上面查看，每一层的神经网络都创建一个name_scope
        n_inputs = int(X.get_shape()[1])  # 特征个数
        stddev = 2 / np.sqrt(n_inputs)
        # 通过指定均值和标准方差来生成正态分布，抛弃那些大于2倍stddev的值。这样将有助于加快训练速度
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        # 权重W不能使用0进行初始化，这样会导致所有的神经元的输出为0，出现对称失效问题，这里使用truncated normal分布(Gaussian)来初始化权重
        W = tf.Variable(init, name='weight')
        b = tf.Variable(tf.zeros([n_neurons]), name='baise')
        y = tf.matmul(X, W) + b
        if activation == "relu":
            return tf.nn.relu(y)
        else:
            return y
