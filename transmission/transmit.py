#!/usr/bin/env python
import argparse as ap
import tensorflow as tf
import configparser
import numpy as np
import os
import functions as f
import scipy as sp
import time
import sys

np.set_printoptions(threshold=np.inf)  # 防止输出太多时被忽略
# constants
co_h = 6.6260657e-34
co_c0 = 299792458
co_lambda = 1550.0e-9
co_dB = 10.0*np.log10(np.exp(1.0))
nu = co_c0/co_lambda
dB_conv = 4.342944819032518


def forward_propagation(P):
    """
    Returns:
        y: received signal (shape = [Nsamp_d, 2], separate real and imaginary part)
        x: symbol vector (shape = [Nsym], complex)
        P: launch power (in W)
    """
    np.random.seed()  # new seed is necessary for multiprocessor
    # P = P_W_r[np.random.randint(P_W_r.shape[0])]  # get random launch power
    # [SOURCE] random points from the signal constellation
    if modulation == "QAM":
        x = const[np.random.randint(const.shape[0], size=[1, Nsym])]
    elif modulation == "Gaussian":
        x = (np.random.normal(0, 1, size=[
             1, Nsym]) + 1j*np.random.normal(0, 1, size=[1, Nsym]))/np.sqrt(2)
    else:
        raise ValueError("wrong modulation format: " + modulation)
    # [MODULATION] upsample + pulse shaping
    x_up = np.zeros([1, Nsamp_a], dtype=np.complex64)
    x_up[:, ::OS_a] = x*np.sqrt(OS_a)
    u = sp.ifft(sp.fft(x_up)*ps_filter_tx_freq)*np.sqrt(P)
    # [CHANNEL] simulate forward propagation
    for NN in range(Nsp):  # enter a span
        for MM in range(fw.model_steps):  # enter a segment
            u = sp.ifft(fw.get_cd_filter_freq(MM)*sp.fft(u))
            u = u*np.exp(1j*fw.nl_param[MM]*np.abs(u)**2)
            #u = u*np.exp(1j*(8/9)*fw.nl_param[MM]*(np.abs(u[0,:])**2+np.abs(u[1,:])**2))
        # add noise, NOTE: amplifier gain (u = u*np.exp(alpha_lin*Lsp/2.0)) is absorbed in nl_param
        u = u + np.sqrt(sigma2/2/Nsp) * (np.random.randn(1,
                                                         Nsamp_a) + 1j*np.random.randn(1, Nsamp_a))
    # [RECEIVER] low-pass filter + downsample
    u = sp.ifft(sp.fft(u)*lp_filter_freq)
    y = u[0, ::OS_a//OS_d]
    y = np.stack([np.real(y), np.imag(y)], axis=1)
    return y, x[0, :], P


# }}}
#========================================================#
# parse function arguments {{{
#========================================================#
parser = ap.ArgumentParser("python3 transmit.py")
parser.description = "Learned Digital Backpropagation (LDBP)"
parser.add_argument(
    "P", help="set of training powers in dB, e.g., [5] or [5,6,7]")
parser.add_argument("Lr", help="learning rate, e.g., 0.01")


parser.add_argument("iter", help="gradient descent iterations, e.g., 1000")
parser.add_argument("-c", "--config_path",
                    help="path to configuration file (default is ldbp_config.ini)", default="ldbp_config.ini")
parser.add_argument(
    "-l", "--logdir", help="directory for log files (default is log)", default="log")
parser.add_argument(
    "-t", "--timing", help="time the forward propagation", action="store_true")

args = parser.parse_args()
args_dict = vars(args)  # converts to a dictionary

opt_list = "P,Lr,iter".split(",")
arg_str = ""
for i in range(len(opt_list)):
    arg_str += opt_list[i]
    arg_str += args_dict[opt_list[i]]
    if(i != len(opt_list)-1):
        arg_str += "_"

config_path = args.config_path
P_dB_r = np.asarray(eval(args.P))
P_W_r = pow(10, P_dB_r/10)*1e-3
iterations = int(args.iter)
learning_rate = float(args.Lr)


# }}}
#========================================================#
# read config file {{{
#========================================================#
defaults = {
    # system
    "sigma scaling": "1",
    "modulation": "16-QAM",
    # LDBP
    "combine half-steps": "yes",
    "load cd filter": "no",
    "load cd filter filename": "parameters.csv",
    "optimize cd filters": "yes",
    "optimize Kerr parameters": "no",
    "complex Kerr parameters": "no",
    "tied Kerr parameters": "no",
    "pruning": "no",
    "less steps than spans": "no",
    "cd alpha": "1",
    "cd filter length margin": "2.0",
    "cd filter length minimum": "13",
    "nl alpha": "1",
    "nl filter length": "1",
    # training
    "adam_A": "0.9",  # decay for running average of the gradient
    "adam_B": "0.999",  # decay for running average of the square of the gradient
    "rmsprop_A": "0.9",
    "rmsprop_B": "0.1",
    "adadelta_A": "0.1",
    "adagrad_A": "0.1",
    # data
    'forward step size method': 'logarithmic',
    'forward split step method': 'symmetric'
}

config = configparser.ConfigParser(defaults)

config_folder, config_file = os.path.split(config_path)
print("configuration file name: '"+config_file+"'")

if not os.path.exists(config_path):
    raise RuntimeError("config file in '"+config_file+"' does not exist")
config.read(config_path)

# system parameters
conf_sys = config['system parameters']
Lsp = conf_sys.getfloat('span length [km]')*1.0e3
alpha = conf_sys.getfloat('alpha [dB/km]')*1.0e-3
gamma = conf_sys.getfloat('gamma [1/W/km]')*1.0e-3
noise_figure = conf_sys.getfloat('amplifier noise figure [dB]')
sigma_scaling = conf_sys.getfloat('sigma scaling')
Nsp = conf_sys.getint('number of spans')
fsym = conf_sys.getfloat('symbol rate [Gbaud]')*1.0e9
modulation = conf_sys['modulation']
rolloff = conf_sys.getfloat('RRC roll-off')
delay = conf_sys.getint('RRC delay')
lp_bandwidth = conf_sys.getfloat('low-pass filter bandwidth [GHz]')*1.0e9
Nsym = conf_sys.getint('data symbols per block')
OS_a = conf_sys.getint('analog oversampling')
OS_d = conf_sys.getint('digital oversampling')
if config.has_option('system parameters', 'D [ps/nm/km]'):
    D = conf_sys.getfloat('D [ps/nm/km]')*1.0e-6
    beta2 = -D*co_lambda**2/(2*np.pi*co_c0)
else:
    beta2 = conf_sys.getfloat('beta2 [ps^2/km]')*1.0e-27


# data generation
conf_data = config['data generation']
StPS_fw = conf_data.getint('forward steps per span')
step_size_method_fw = conf_data['forward step size method']
ssfm_method_fw = conf_data['forward split step method']

if OS_a % OS_d != 0:
    raise ValueError(
        'oversampling factors have to be divisible: OS_a={}, OS_d={}'.format(OS_a, OS_d))


# derived parameters
L = Lsp*Nsp
Gain = 10.0**(alpha*Lsp/10.0)
sef = 10.0**(noise_figure/10.0)/2.0/(1.0-1.0/Gain)
alpha_lin = alpha / dB_conv
N0 = sigma_scaling*Nsp*(np.exp(alpha_lin*Lsp)-1.0)*co_h*nu*sef
sigma2 = N0 * fsym * OS_a
Nsamp_a = Nsym*OS_a
Nsamp_d = Nsym*OS_d
fsamp_a = fsym*OS_a
fsamp_d = fsym*OS_d
f_a = f.get_fvec(Nsamp_a, fsamp_a)
f_d = f.get_fvec(Nsamp_d, fsamp_d)

if "QAM" in modulation:
    splitstr = modulation.split("-")
    modulation_order = int(splitstr[0])
    modulation = "QAM"
# }}}
#========================================================#
# forward propagation generative model {{{
#========================================================#
ps_filter_tx_coeffs = f.rrcosine(rolloff, delay, OS_a)  # pulse shaping filter
ps_filter_tx_length = 2*(OS_a*delay)+1
ps_filter_tx_delay = OS_a*delay  # delay in samples

# pre-compute frequency responses
ps_tmp = np.concatenate(
    (ps_filter_tx_coeffs, np.zeros(Nsamp_a-ps_filter_tx_length)))
ps_tmp = np.roll(ps_tmp, -ps_filter_tx_delay)
ps_filter_tx_freq = sp.fft.fft(ps_tmp, n=Nsamp_a)
lp_filter_freq = (abs(f_a) <= lp_bandwidth/2).astype(float)

if modulation == "QAM":
    const = f.QAM(modulation_order)

ssfm_opts = {
    "alpha": alpha,
    "beta2": beta2,
    "gamma": gamma,
    "Nsp": 1,
    "Lsp": Lsp,
    "fsamp": fsamp_a,
    "Nsamp": Nsamp_a,
    "step_size_method": step_size_method_fw,
    "ssfm_method": ssfm_method_fw,
    "StPS": StPS_fw,
    "direction": 1
}

fw = f.ssfm_parameters(ssfm_opts)

# 隐藏层函数


def invisible_layer(inputs, in_size, out_size, keep_prob=1.0, activation_function=None):
    Weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
    biases = tf.Variable(tf.zeros([out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
        outputs = tf.nn.dropout(outputs, keep_prob)  # 随机失活
    return outputs


# 将结果写入文件
f = open("result.txt", "w")
for P_dB in np.nditer(P_dB_r):
    P_W = pow(10, P_dB/10)*1e-3
    print("")
    print("timing the forward propation ...", file=f)
    t = time.time()
    y_d, x_d, p = forward_propagation(P_W)
    # 将结果输出到文件夹
    print("y: ", y_d, file=f)
    print("x: ", x_d, file=f)
    print("p: ", p, file=f)
    # 调用network，跑tensorflow方法
    sess = tf.InteractiveSession()
    # holder变量
    x_h = tf.placeholder(tf.float32, [None, 784])
    y_h = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)     # 概率

    for i in range(64):
        h1 = invisible_layer(x_d, 784, 300, keep_prob, tf.nn.relu)
        x_d = invisible_layer(h1, 784, 300, keep_prob, tf.nn.relu)

    # 输出层
    w = tf.Variable(tf.zeros([300, 10]))  # 300*10
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(h1, w)+b)

    # 定义loss,optimizer
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_h *
                                                  tf.log(y), reduction_indices=[1]))
    train_step = tf.train.AdagradOptimizer(0.35).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_h, 1))  # 高维度的
    acuracy = tf.reduce_mean(
        tf.cast(correct_prediction, tf.float32))  # 要用reduce_mean

    tf.global_variables_initializer().run()
    for i in range(3000):  # 3000是循环次数
        batch_x, batch_y = y_d.next_batch(100)
        train_step.run({x_h: batch_x, y_h: batch_y, keep_prob: 0.75})
        if i % 1000 == 0:
            train_accuracy = acuracy.eval(
                {x_h: batch_x, y_h: batch_y, keep_prob: 1.0})
            print("step %d,train_accuracy %g" % (i, train_accuracy))

    elapsed = time.time()-t
    print("{0:.2f} seconds to generate 1 input/output data pair".format(elapsed), file=f)
    sys.exit("")
