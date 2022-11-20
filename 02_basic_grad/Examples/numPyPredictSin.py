import math
import numpy as np


x=np.linspace(-math.pi,math.pi,2000)# generate a sequence with 2000 numbers with same pace
y=np.sin(x)# get real answer of sin function

print(x[1:10]);

#assume the function is y=ax^3+bx^2+cx+d

#initialize the function
a=np.random.randn()
b=np.random.randn()
c=np.random.randn()
d=np.random.randn()

learning_rate=1e-6
learning_times=10000


for itime in range(learning_times):
    y_predict=a*(x**3)+b*(x**2)+c*x+d
    y_delta=y_predict-y

    #loss=np.square(y_delta).sum()
    #according to 求导结果
    grad_a=(2*y_delta*(x**3)).sum() # a对损失函数的贡献梯度
    grad_b=(2*y_delta*(x**2)).sum()
    grad_c=(2*y_delta*x).sum()
    grad_d=(2*y_delta).sum()

    a-=learning_rate*grad_a
    b-=learning_rate*grad_b
    c-=learning_rate*grad_c
    d-=learning_rate*grad_d

print(a)
print(b)
print(c)
print(d)

