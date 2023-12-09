from pylab import *


x_axis_data = [ 0.14 , 0.29 , 0.29 , 0.43 , 0.57 , 0.71 , 0.71 , 0.86 , 0.86 , 1.00]
y_axis_data = [ 1.00 , 1.00 , 0.67 , 0.75 , 0.8  , 0.83 , 0.71 , 0.75 , 0.67 , 0.7]

plt.title('PR-Curve') 
# plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
plt.plot(x_axis_data, y_axis_data, 'ro-', color='b', alpha=1.0, linewidth=1., label='rank')

plt.axis([0.0,1.05, 0.0,1.05])
plt.xticks([i * 0.1 for i in range(0, 11)]) ## 显示的x轴刻度值
plt.yticks([i * 0.1 for i in range(0, 11)])   ## 显示y轴刻度值
# 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
plt.legend(loc="upper right")
plt.xlabel('Recall')
plt.ylabel('Precision')

plt.show()
# plt.savefig('demo.jpg')  # 保存该图片
