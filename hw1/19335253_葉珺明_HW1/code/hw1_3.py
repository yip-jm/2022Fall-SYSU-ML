import matplotlib.pyplot as plt
import numpy as np
import math
import random
from mpl_toolkits.mplot3d import Axes3D
N = 100
V = 4000000.0

rdom_pnt = [10,20,30,40,50,60,70,80,100,200,500]
res = np.zeros(len(rdom_pnt))

def f(x,y):
    z = y**2*np.exp(x**2-y**2)/x+x**3
    return z

def bool_in_func(z1, z2):
    if z1 >= z2:
        return 1
    else:
        return 0

def draw_pic(inx, outx, iny, outy, inz, outz):
    fig = plt.figure()

    axes = Axes3D(fig,auto_add_to_figure=False)
    fig.add_axes(axes)

    x = np.arange(2, 4.01, 0.02)
    y = np.arange(-1, 1.01, 0.02)#前两个参数为自变量取值范围
    x, y = np.meshgrid(x,y)

    z = y**2*np.exp(x**2-y**2)/x+x**3
    # print(np.max(np.max(z)))
    axes.plot_surface(x,y,z,cmap='Wistia')

    axes.scatter(inx, iny, inz, s=16., color='r')
    axes.scatter(outx, outy, outz, s=16., color='b')

    plt.show()

print('Random points：')
print("-------------------ONLY 1---------------------")
for i in range (len(rdom_pnt)):
    x_in_list = []
    y_in_list = []
    x_out_list = []
    y_out_list = []
    z_in_list = []
    z_out_list = []
    mysum = 0
    inn = rdom_pnt[i]
    for j in range (inn):
        x = random.uniform(2.0, 4.0)
        y = random.uniform(-1.0, 1.0)
        z = random.uniform(0, 1000000)

        if bool_in_func(f(x, y), z):
            x_in_list.append(x)
            y_in_list.append(y)
            z_in_list.append(z)
            mysum += 1
        else:
            x_out_list.append(x)
            y_out_list.append(y)
            z_out_list.append(z)

    draw_pic(x_in_list,x_out_list,y_in_list,y_out_list,z_in_list,z_out_list)
    in_func_pnt = mysum/inn
    res[i] = in_func_pnt*V
res = np.around(res,3)
print('res : ', res)

print("--------------REPEAT 100 TIMES----------------")
all_res = np.zeros((N, len(rdom_pnt)))
for n in range (N):
    for i in range (len(rdom_pnt)):
        mysum = 0
        inn = rdom_pnt[i]
        for j in range (inn):
            x = random.uniform(2.0, 4.0)
            y = random.uniform(-1.0, 1.0)
            z = random.uniform(0, 1000000)        

            if bool_in_func(f(x, y), z):
                mysum += 1
    
        in_func_pnt = mysum/inn
        res[i] = in_func_pnt*V
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
    z_list = []
    x_out_list = []
    y_out_list = []
    z_out_list = []
    myvol = 0
    inn = rdom_pnt[i]
    for j in range (inn):
        x = random.uniform(2.0, 4.0)
        y = random.uniform(-1.0, 1.0)
        z = f(x, y)

        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
        myvol += z*(4.0-2.0)*2.0
           
    draw_pic(x_list,x_out_list,y_list,y_out_list,z_list,z_out_list)
    avg_vol = myvol/inn
    res[i] = avg_vol

print('res : ', np.around(res,3))

print("--------------REPEAT 100 TIMES----------------")
all_res = np.zeros((N, len(rdom_pnt)))
for n in range (N):
    for i in range (len(rdom_pnt)):
        myvol = 0
        inn = rdom_pnt[i]
        for j in range (inn):
            x = random.uniform(2.0, 4.0)
            y = random.uniform(-1.0, 1.0)
            z = f(x, y)
            myvol += z*(4.0-2.0)*2.0
        avg_vol = myvol/inn
        res[i] = avg_vol
    all_res[n] = res

mean = np.mean(all_res, axis=0)
print('mean: ', np.around(mean, 3))

var = np.var(all_res, axis=0)
print('var : ', np.around(var, 3))
