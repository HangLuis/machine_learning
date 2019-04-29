#SGD随机梯度下降算法
from matplotlib import pyplot as plt
import random


#生成数据
def data():
    x = range(40)
    y = [(3 * i + 2) for i in x]
    for i in range(40):
        y[i] = y[i] + random.randint(-20, 20)  #添加噪声
    return x, y


#使用随机梯度下降训练
def SGD(x, y):
    error0 = 0
    step_size = 0.001  #学习率
    esp = 1e-5  #误差阈值
    #给a，b随机赋初始值
    a = random.randint(0, 4)
    b = random.randint(0, 4)
    m = len(x)
    n = 1
    while True:
        n = n + 1
        if (n > 2000):
            break
        print('第%d次迭代' % n)
        i = random.randint(0, m - 1)  #随机选取一个样本
        print('选取第%d组样本' % i)
        sum0 = a * x[i] + b - y[i]
        sum1 = (a * x[i] + b - y[i]) * x[i]
        error1 = (a * x[i] + b - y[i])**2  #代价函数
        a = a - sum1 * step_size / m
        b = b - sum0 * step_size / m
        print('a=%f,b=%f' % (a, b))
        print("\n")
        if abs(error1 - error0) < esp:  #误差小于阈值时可以认为趋于平缓
            break
        error0 = error1
    return a, b


#程序入口
if __name__ == '__main__':
    x, y = data()
    a, b = SGD(x, y)
    X = range(40)
    Y = [(a * i + b) for i in X]
    plt.scatter(x, y, color='red')
    plt.plot(X, Y)
    plt.show()