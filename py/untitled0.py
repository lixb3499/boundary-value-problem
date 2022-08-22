# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 21:47:18 2022

@author: lixb
"""

import numpy as np
import matplotlib.pyplot as plt

# 随机X纬度x1，rand是随机均匀分布
x = 2 * np.random.rand(100,1)
# 人为设置真实的Y一列 np.random.randn(100,1) 设置误差遵循标准正态分布
y = 4 + 3 * x + np.random.randn(100,1)
# 整合 x0 和 x1 成矩阵
x_b = np.c_[np.ones((100,1)),x]

learning_rate = 0.01 # 学习率一般默认设置为0.01
n_iterations = 10000 # 迭代次数够多即可(不一定需要全局最优解)
m = 100 # 100 行

# #1 初始化theta，w0...wn
theta = np.random.randn(2,1) # x_b 中只有x0,x1,只需要两个theta

# #4 不设置阀值，直接设置超参数，迭代次数，迭代次数到了就认为收敛了
for iteration in range(n_iterations):
    # #2 求梯度gradient
    index = np.random.randint(m) # 随机索引，抽取出来，进行训练
    xi = x_b[index:index+1] # 抽取随机x
    yi = y[index:index+1] # 抽取随机x对应的y
    gradients = x_b.T.dot(x_b.dot(theta)-y)# 不需要1/m平权
    # #3 调整theta值
    theta = theta - learning_rate * gradients

print(theta)

x_new = np.array([[0],[2]])
x_new_b = np.c_[(np.ones((2,1))),x_new]
y_predict = x_new_b.dot(theta)

# 绘制图形
plt.plot(x_new,y_predict,'r-')
plt.plot(x,y,'b.')
plt.axis([0,2,0,15])
plt.show()

