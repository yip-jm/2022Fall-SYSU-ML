import numpy as np
import pandas as pd
from random import randint
from pylab import *

alpha = 0.00001
iters = 15000001

def get_data(filepath):
   
    df = pd.read_csv(filepath, sep='\\s+', header = None, names = ['Square', 'Dis', 'Price'], )
    df.head()

    x = df.loc[:, ['Square', 'Dis']]
    x.insert(0, 'ones', 1)
    y = df.loc[:, ['Price']]

    x = np.array(x)
    y = np.array(y)

    return x,y

def loss_func(x, y, theta):
    loss = np.sum(np.power(np.dot(theta, x.T)-y.T, 2))
    return loss

def GradientDescent(x, y, theta, alpha, iters):
    train_ = []
    test_ = []

    test = r"/home/yip/machine_learning/hw3/data/dataForTestingLinear.txt"
    test_x, test_y = get_data(test)

    for i in range(iters):
        r = randint(0, 49) 
        t0 = np.sum(np.dot(theta, x[r].T)-y[r].T)
        t1 = np.sum((np.dot(theta, x[r].T)-y[r].T)*x[r].T[1])
        t2 = np.sum((np.dot(theta, x[r].T)-y[r].T)*x[r].T[2])
        new_a = np.array([t0, t1, t2])
        
        theta = theta - alpha*new_a
        if i!=0 and i%1000000 == 0:
            print(theta)
            train_loss = loss_func(x, y, theta)
            train_.append(train_loss)

            test_loss = loss_func(test_x, test_y, theta)
            test_.append(test_loss)
            
    return theta, train_, test_

def draw_pic(train_Loss, test_Loss):
    
    x_ = []
    for i in range(15):
        x_.append(i+1)

    plt.xticks([i for i in range(0, 16)])
    plt.title('Gradient Descent') 

    plt.plot(x_, train_Loss, 'ro-', color='g', alpha=1.0, linewidth=1.2, label='Train')
    plt.plot(x_, test_Loss, 'ro-', color='b', alpha=1.0, linewidth=1.2, label='Test')
    
    plt.legend(loc="upper right")
    plt.xlabel('iters(e06)')
    plt.ylabel('Loss')
    
    plt.show()



train = r"/home/yip/machine_learning/hw3/data/dataForTrainingLinear.txt"
x,y = get_data(train)
theta = np.array([0,0,0])

theta, train_Loss, test_Loss = GradientDescent(x, y, theta, alpha, iters)
draw_pic(train_Loss, test_Loss)

print(train_Loss)
print(test_Loss)

