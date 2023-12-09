import numpy as np
import pandas as pd
from random import randint
import math
from pylab import *

alpha = 0.0001
iters = 10000
E = math.e
def get_data(k, filepath):
   
    df = pd.read_csv(filepath, sep='\\s+', header = None, names = ['1', '2', '3', '4', '5', '6', '7'], )
    df.head()

    x = df.loc[0:k, ['1', '2', '3', '4', '5', '6']]
    x.insert(0, 'ones', 1)
    y = df.loc[0:k, ['7']]

    x = np.array(x)
    y = np.array(y)
    y = np.squeeze(y)
    return x,y


def error_rate(true_y, pred_y):
    cnt = 0.0
    for i in range (len(true_y)):
        if true_y[i] != pred_y[i]:
            cnt += 1
    cnt /= len(true_y)
    return cnt

def classify(theta, x):
    test_y = []
    for i in range (len(x)):
        s = 1/(1+E**(-(np.sum(np.dot(theta, x[i].T)))))
        if s>= 0.5:
            test_y.append(1)
        else:
            test_y.append(0)

    return test_y

def GradientDescent(k, x, y, theta, alpha, iters):

    for i in range(iters):
        r = randint(0, k-1) 
        down = 1/(1+E**(-(np.sum(np.dot(theta, x[r].T)))))
        # print(down)
        t0 = y[r].T - down
        t1 = (y[r].T - down)*x[r].T[1]
        t2 = (y[r].T - down)*x[r].T[2]
        t3 = (y[r].T - down)*x[r].T[3]
        t4 = (y[r].T - down)*x[r].T[4]
        t5 = (y[r].T - down)*x[r].T[5]
        t6 = (y[r].T - down)*x[r].T[6]


        new_a = np.array([t0, t1, t2, t3, t4, t5, t6])
        theta = theta + alpha*new_a

    return theta

def draw_pic(k, tn_er, tt_er):

    # plt.xticks([i*50 for i in range(0, 11)])
    plt.title('Misclassified on both') 

    plt.plot(k, tn_er, 'r-', alpha=1.0, linewidth=1.2, label='Train_Error_Rate')
    plt.plot(k, tt_er, 'b-', alpha=1.0, linewidth=1.2, label='Test_Error_Rate')
    
    plt.legend(loc="upper right")
    plt.xlabel('iters')
    plt.ylabel('Error_Rate')
    
    plt.show()



train = r"/home/yip/machine_learning/hw3/data/dataForTrainingLogistic.txt"
test =  r"/home/yip/machine_learning/hw3/data/dataForTestingLogistic.txt"

k = []
tn_er = []
tt_er = []
for i in range (1, 41):

    k.append(i*10)

    theta = np.zeros(7)

    train_x, train_y = get_data(i*10, train)
    test_x, test_y = get_data(100, test)

    theta = GradientDescent(i*10, train_x, train_y, theta, alpha, iters)

    train_pred_y = classify(theta, train_x)
    train_er = error_rate(train_y, train_pred_y)
    tn_er.append(train_er)

    test_pred_y = classify(theta, test_x)
    test_er = error_rate(test_y, test_pred_y)
    tt_er.append(test_er)


draw_pic(k, tn_er, tt_er)
















# print(train_Loss)
# print(test_Loss)
# draw_pic(train_Loss, test_Loss)



