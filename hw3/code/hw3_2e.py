import numpy as np
import pandas as pd
from random import randint
import math
from pylab import *

alpha = 0.0001
iters = 10001
E = math.e
def get_data(filepath):
   
    df = pd.read_csv(filepath, sep='\\s+', header = None, names = ['1', '2', '3', '4', '5', '6', '7'], )
    df.head()

    x = df.loc[:, ['1', '2', '3', '4', '5', '6']]
    x.insert(0, 'ones', 1)
    y = df.loc[:, ['7']]

    x = np.array(x)
    y = np.array(y)
    y = np.squeeze(y)
    return x,y

def log_loss(x, y, theta):
    loss = 0
    for i in range (len(x)):
        wx  = np.sum(np.dot(theta, x[i].T))
        tmp = wx*y[i] - math.log(1+E**wx, E)
        loss = loss + tmp
    return loss

def draw_pnt(n, test_y, test_y_pred):
    x_ = []
    for i in range(n):
        x_.append(i+1)

    plt.xticks([i*10 for i in range(0, 11)])
    
    plt.title('TRUE and Predict')

    plt.ylim(-0.5, 1.5)
    plt.hlines(0.5, 0, 101, color='gray')

    plt.scatter(x_, test_y, s=20, color='r', marker='o', label='True')
    plt.scatter(x_, test_y_pred, s=30, color='b', marker='x', label='Predict')

    plt.legend(loc="upper right")
    plt.xlabel('Testing Data i')
    plt.ylabel('True/Predict')
    plt.show()





def compara(true_y, pred_y):
    cnt = 0
    for i in range (len(true_y)):
        if true_y[i] != pred_y[i]:
            cnt += 1
    
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

def GradientDescent(x, y, theta, alpha, iters):
    train_ = []
    test_ = []
    cnt = []

    test = r"/home/yip/machine_learning/hw3/data/dataForTestingLogistic.txt"
    test_x, test_y = get_data(test)

    for i in range(iters):
        r = randint(0, 399) 
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
        test_y_pred = classify(theta, test_x)

        cnt.append(compara(test_y, test_y_pred))


        # if i!=0 and i%1000000 == 0:
        #     print(theta)
        #     train_loss = log_loss(x, y, theta)
        #     train_.append(train_loss)

        #     test_loss = log_loss(test_x, test_y, theta)
        #     test_.append(test_loss)
    
    

    # draw_pnt(len(test_y), test_y, test_y_pred)
    draw_miss(cnt)

    return theta, train_, test_

def draw_miss(cnt):

    x_ = []
    for i in range(len(cnt)):
        x_.append(i+1)

    plt.xticks([i*1000 for i in range(0, 11)])
    plt.title('Misclassified') 
    plt.ylim((0, 60))
    plt.plot(x_, cnt, 'b-', alpha=1.0, linewidth=1.2, label='cnt')

    plt.legend(loc="upper right")
    plt.xlabel('iters')
    plt.ylabel('Misclassified Count')

    plt.show()
    



def draw_pic(train_Loss, test_Loss):
    
    x_ = []
    for i in range(50):
        x_.append(i+1)

    plt.xticks([i*5 for i in range(0, 11)])
    plt.title('Gradient Descent') 

    plt.plot(x_, train_Loss, 'g-', alpha=1.0, linewidth=1.2, label='Train')
    plt.plot(x_, test_Loss, 'b-', alpha=1.0, linewidth=1.2, label='Test')
    
    plt.legend(loc="lower right")
    plt.xlabel('iters(e06)')
    plt.ylabel('Loss')
    
    plt.show()



train = r"/home/yip/machine_learning/hw3/data/dataForTrainingLogistic.txt"
x,y = get_data(train)
theta = np.zeros(7)


theta, train_Loss, test_Loss = GradientDescent(x, y, theta, alpha, iters)
print(train_Loss)
print(test_Loss)
# draw_pic(train_Loss, test_Loss)


