import random
import matplotlib.pyplot as plt
import numpy as np
import math

def bool_in_r(x, y):
    if x**2+y**2 <= 1:
        return 1
    else:
        return 0

def draw_pic(inx, iny, outx, outy):
    rx = np.arange(0,1.01,0.01)
    # print(rx)
    ry = []
    for x0 in rx:
        ry.append(math.sqrt(1-x0*x0))
    
    plt.scatter(inx, iny, s=16., color='g')
    plt.scatter(outx, outy, s=16., color='b')
    plt.plot(rx, ry, color='r')
    plt.xlim(0,1.01)
    plt.ylim(0,1.01)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


rdom_pnt = [20, 50, 100, 200, 300, 500, 1000, 5000]
pii = np.zeros(len(rdom_pnt))

print("-------------------ONLY 1---------------------")
for i in range (len(rdom_pnt)):
    x_in_list = []
    y_in_list = []
    x_out_list = []
    y_out_list = []
    inn = rdom_pnt[i]
    mysum = 0
    for j in range (inn):
        x = random.random()
        y = random.random()
        # print(i, 'x: ', x, 'y: ', y)
        if (bool_in_r(x,y)):
            x_in_list.append(x)
            y_in_list.append(y)
            mysum += 1
        else:
            x_out_list.append(x)
            y_out_list.append(y)
    draw_pic(x_in_list,y_in_list,x_out_list,y_out_list)
    in_r_pnt = mysum/inn
    pi = in_r_pnt*4
    pii[i] = pi
    
print('res: ', np.around(pii,3))

print('')
print("--------------REPEAT 100 TIMES----------------")
all_pii = np.zeros((100, len(rdom_pnt)))
for n in range (100):
    for i in range (len(rdom_pnt)):
        x_in_list = []
        y_in_list = []
        inn = rdom_pnt[i]
        mysum = 0
        for j in range (inn):
            x = random.random()
            y = random.random()
            if (bool_in_r(x,y)):
                x_in_list.append(x)
                y_in_list.append(y)
                mysum += 1

        in_r_pnt = mysum/inn
        pi = in_r_pnt*4
        pii[i] = pi
    all_pii[n] = pii

mean = np.mean(all_pii, axis=0)
print('mean: ', np.around(mean, 3))

var = np.var(all_pii, axis=0)
print('var : ', np.around(var, 3))
