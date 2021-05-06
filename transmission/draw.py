
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
def draw_loss(filename):
    """

    """
    file=open(filename,'r')
    loss=[]
    for line in file:
        line=line.strip('\n')
        if "Loss:" in line:
            line.find(":")
            loss.append(eval(line[line.find("Loss:")+5:]))
    print(loss)
    plt.plot(np.linspace(9,1999,200),loss)
    plt.xlabel('训练迭代次数')
    plt.ylabel('loss')
    plt.show()

def draw_train_accuracy(filename):
    """

    """
    file=open(filename,'r')
    train_acc=[]
    for line in file:
        line=line.strip('\n')
        if "Train Accuracy:" in line:
            train_acc.append(eval(line[line.find(":")+1:]))
    print(train_acc)
    plt.plot(np.linspace(9,1999,200),train_acc)
    # # plt.xlabel('训练迭代次数')
    plt.ylabel('train accuracy')
    plt.show()

def draw_test_accuracy(filename):
    """

    """
    file = open(filename, 'r')
    train_acc = []
    for line in file:
        line = line.strip('\n')
        if "Test Accuracy:" in line:
            train_acc.append(eval(line[line.find(":") + 1:]))
    print(train_acc)
    plt.plot(np.linspace(9, 1999, 200), train_acc)
    # # plt.xlabel('训练迭代次数')
    plt.ylabel('test accuracy')
    plt.show()

# draw_test_accuracy('record.txt')
def drawBer():
    # imag_accuracy=[0.247833251953125,0.24688720703125,0.963836669921875,0.974395751953125,0.982757568359375,
    #                0.98431396484375,0.254119873046875,0.250701904296875,0.95196533203125]
    # real_accuracy=[0.24809276777540434,0.2462618248397925,0.9673481843149222,0.9732478893296714,
    #                0.9829111992676228,0.9839283897874072,0.25094090123080054,0.2511443393347574,
    #                0.9557522123893806]
    p=[-4,-3,-2,-1,0,1,2,3,4]
    ber_y=[0.03800201416015625,0.0260009765625,0.0160369873046875,0.4993743896484375,0.5020599365234375,0.500732421875,
         0.009185791015625,0.01432037353515625,0.49994659423828125]
    ber_x=[0.49799346923828125,0.4990234375,0.01839447021484375,0.01300048828125,0.008758544921875,
           0.0079803466796875,0.49700927734375,0.4951324462890625,0.02342987060546875]
    standard_y=[0.041519,0.02887,0.01825,0.013779,0.010437,0.0093765,0.011566,0.016541,0.028015]
    standard_x=[0.043793,0.031723,0.02079,0.015144,0.0099869,0.0092087,0.0096207,0.015182,0.026169]
    real_imag_x=[0.49627685546875,0.49399566650390625,0.01804351806640625,0.013214111328125,0.008941650390625,0.00768280029296875,
                 0.49427032470703125,0.49530792236328125,0.02344512939453125,]
    real_imag_y=[0.532318115234375,0.519256591796875,0.01513671875,0.49881744384765625,
                 0.49854278564453125,0.49884033203125,0.5270233154296875,0.5056381225585938,0.5002593994140625]

    #剔除异常值
    ber_y = [0.03800201416015625, 0.0260009765625, 0.0160369873046875,
             0.009185791015625, 0.01432037353515625]
    ber_x = [ 0.01839447021484375, 0.01300048828125, 0.008758544921875,
             0.0079803466796875,   0.02342987060546875]
    standard_y = [0.041519, 0.02887, 0.01825,  0.011566, 0.016541]
    standard_x = [ 0.02079, 0.015144, 0.0099869, 0.0092087,   0.026169]
    real_imag_x=[0.01804351806640625,0.013214111328125,0.008941650390625,0.00768280029296875,
                 0.02344512939453125,]
    real_imag_y = [0.01513671875]

    p=[-4,-3,-2,2,3]
    po=[-2]

    plt.plot(p,ber_y,'b',label='real_y_ber',marker='o')
    plt.plot(p,standard_y,'r',label='dbp_y_ber',marker='o')
    plt.plot(po,real_imag_y,'g',label='real_imag_y_ber',marker='o')
    plt.legend()
    plt.ylabel("ber_y")
    plt.xlabel("P/dBm")
    plt.show()

def drawAcc():
    p = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    accuracy_y_imag=[0.925933837890625,0.9482421875,0.9674072265625,0.24615478515625,0.249267578125,0.24468994140625,0.981201171875,0.97088623046875,0.243499755859375]
    accuracy_y_real=[0.92205810546875,0.94775390625,0.96844482421875,0.25433349609375,0.24774169921875,0.252716064453125,0.9825439453125,0.971893310546875,0.250640869140625]
    plt.plot(p, accuracy_y_imag, 'b', label='accuracy_y_imag')
    plt.plot(p, accuracy_y_real, 'r', label='accuracy_y_real')
    plt.legend()
    plt.ylabel("accuracy_y")
    plt.xlabel("P/dBm")
    plt.show()


drawBer()