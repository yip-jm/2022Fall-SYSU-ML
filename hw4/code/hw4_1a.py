import math
import numpy as np
from pylab import *

er = 1e-3

D = [[6.2, 3.2], [6.6, 3.7], [6.5, 3.0]]
D = np.array(D)
X = [[5.9, 3.2], [4.6, 2.9], [6.2, 2.8], [4.7, 3.2], [5.5, 4.2], 
    [5.0, 3.0], [4.9, 3.1], [6.7, 3.1], [5.1, 3.8], [6.0, 3.0]]
X = np.array(X)

def draw_pic(C0, C1, C2, D):

    plt.scatter(D[0][0], D[0][1], color='r', marker='*', label='Centroid 1')
    plt.scatter(D[1][0], D[1][1], color='g', marker='*', label='Centroid 2')
    plt.scatter(D[2][0], D[2][1], color='b', marker='*', label='Centroid 3')
    
    x0 = []
    y0 = []
    for i in range (len(C0)):
        x0.append(C0[i][0])
        y0.append(C0[i][1])
    
    plt.scatter(x0, y0, color='r', marker='o', label='Cluster 1')

    x1 = []
    y1 = []
    for i in range (len(C1)):
        x1.append(C1[i][0])
        y1.append(C1[i][1])
    
    plt.scatter(x1, y1, color='g', marker='x', label='Cluster 2')

    x2 = []
    y2 = []
    for i in range (len(C2)):
        x2.append(C2[i][0])
        y2.append(C2[i][1])
    
    plt.scatter(x2, y2, color='b', marker='^', label='CLuster 3')
    plt.legend(loc="upper right")
    plt.title('K-Means')
    
    plt.xlim((4.3, 6.8))
    plt.ylim((2.6, 4.4))
    plt.show()


def update_C(prev_D):
    new_D = np.mean(prev_D, 0)
    return new_D.tolist()

n = 0
while(1):
    r = []
    g = []
    b = []
    for i in range (len(X)):
        d0 = math.sqrt(np.sum((X[i]-D[0])**2))
        d1 = math.sqrt(np.sum((X[i]-D[1])**2))
        d2 = math.sqrt(np.sum((X[i]-D[2])**2))

        if d0 >= d1:
            if d1 >= d2:
                b.append(X[i])
            else:
                g.append(X[i])
        elif d0 >= d2:
            if d2 >= d1:
                g.append(X[i])
            else:
                b.append(X[i])
        else:
            r.append(X[i])
        
    new_D0 = update_C(r)
    new_D1 = update_C(g)
    new_D2 = update_C(b)
    if math.fabs(math.sqrt(np.sum(new_D0-D[0])**2)) >= er \
        or math.fabs(math.sqrt(np.sum(new_D1-D[1])**2)) >= er\
            or math.fabs(math.sqrt(np.sum(new_D2-D[2])**2)) >= er:
        
        n += 1
        D[0] = new_D0
        D[1] = new_D1
        D[2] = new_D2
        draw_pic(r, g, b, D)
        print(D)
    else:
        print(D)
        break


print(n)
