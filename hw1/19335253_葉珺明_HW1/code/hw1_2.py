import random
import matplotlib.pyplot as plt
import numpy as np
import math
N = 100

def bool_in_func(x, y):
    if x*x*x >= y:
        return 1
    else:
        return 0

def draw_pic(inx, iny, outx, outy):
    fx = np.arange(0,1.01,0.01)

    fy = []
    for x0 in fx:
        fy.append(x0*x0*x0)
    
    plt.scatter(inx, iny, s=16., color='g')
    plt.scatter(outx, outy, s=16., color='b')
    plt.plot(fx, fy, color='r')
    plt.xlim(0,1.01)
    plt.ylim(0,1.01)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()

rdom_pnt = [5,10,20,30,40,50,60,70,80,90,100]
res = np.zeros(len(rdom_pnt))

print('Random points：')
print("-------------------ONLY 1---------------------")
for i in range (len(rdom_pnt)):
    x_in_list = []
    y_in_list = []
    x_out_list = []
    y_out_list = []
    mysum = 0
    inn = rdom_pnt[i]
    for j in range (inn):
        x = random.random()
        y = random.random()

        if bool_in_func(x, y):
            x_in_list.append(x)
            y_in_list.append(y)
            mysum += 1
        else:
            x_out_list.append(x)
            y_out_list.append(y)      
    draw_pic(x_in_list,y_in_list,x_out_list,y_out_list)
    in_func_pnt = mysum/inn
    res[i] = in_func_pnt

print('res : ', np.around(res,3))

print("--------------REPEAT 100 TIMES----------------")
all_res = np.zeros((N, len(rdom_pnt)))
for n in range (N):
    for i in range (len(rdom_pnt)):
        x_in_list = []
        y_in_list = []
        x_out_list = []
        y_out_list = []
        mysum = 0
        inn = rdom_pnt[i]
        for j in range (inn):
            x = random.random()
            y = random.random()

            if bool_in_func(x, y):
                mysum += 1
     
        in_func_pnt = mysum/inn
        res[i] = in_func_pnt
    all_res[n] = res

mean = np.mean(all_res, axis=0)
print('mean: ', np.around(mean, 3))

var = np.var(all_res, axis=0)
print('var : ', np.around(var, 3))

print('')
print('Mathematical expectation value： ')
print("-------------------ONLY 1---------------------")
for i in range (len(rdom_pnt)):
    x_list = []
    y_list = []
    x_out_list = []
    y_out_list = []
    mysquare = 0
    inn = rdom_pnt[i]
    for j in range (inn):
        x = random.random()
        y = x*x*x

        x_list.append(x)
        y_list.append(y)
        mysquare += y*(1-0)
           
    draw_pic(x_list,y_list,x_out_list,y_out_list)
    avg_sq = mysquare/inn
    res[i] = avg_sq

print('res : ', np.around(res,3))

print("--------------REPEAT 100 TIMES----------------")
all_res = np.zeros((N, len(rdom_pnt)))
for n in range (N):
    for i in range (len(rdom_pnt)):
        mysquare = 0
        inn = rdom_pnt[i]
        for j in range (inn):
            x = random.random()
            y = x*x*x
            mysquare += y*(1-0)
        avg_sq = mysquare/inn
        res[i] = avg_sq
    all_res[n] = res

mean = np.mean(all_res, axis=0)
print('mean: ', np.around(mean, 3))

var = np.var(all_res, axis=0)
print('var : ', np.around(var, 3))
