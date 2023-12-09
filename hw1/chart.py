import matplotlib.pyplot as plt
import numpy as np

x = [10  ,  20  ,  30  ,  40  ,  50  ,  60  ,  70  ,  80 ,  100  , 200, 500 ]
y1 =[1.090,1.120,1.151,1.131,1.147,1.154,1.078,1.147,1.114,1.132,1.148]
y2 = [12.00,5.897,3.949,3.406,2.698,2.042,1.886,1.608,1.266,0.746,0.217]
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(x, y1, 'o', color='b')
ax1.plot(x, y1, '-', color='b')
ax1.set_ylabel('MEAN(e+5)',color='b')
ax1.set_title("Integral-2")
plt.tick_params(axis='y',colors='b')

ax2 = ax1.twinx()  # this is the important function
ax2.plot(x, y2, '--',color='r')
ax2.plot(x, y2, 's',color='r')
ax2.set_xlim([0, 500])
ax2.set_ylabel('VAR(e+9)', color='r')
plt.tick_params(axis='y',colors='r')
ax1.set_xlabel('X -- Random Points')

plt.show()
33