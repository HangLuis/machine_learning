#MBGD小批量梯度下降算法
from matplotlib import pyplot as plt
import random


#生成数据
def data():
    x = range(40)
    y = [(3 * i + 2) for i in x]
    for i in range(len(y)):
        y[i] = y[i] + random.randint(-20, 20)
    return x, y


#用小批量梯度下降算法进行迭代
def MBGD(x, y):
    error0 = 0
    error1 = 0
    n = 0
    m = len(x)
    esp = 1e-5
    step_size = 0.001  #选择合理的步长
    a = random.randint(0, 10)  #给a，b赋初始值
    b = random.randint(0, 10)
    while True:
        trainList = []
        for i in range(5):  #创建随机的批量
            trainList.append(random.randint(0, m - 1))
        for i in range(5):  #对数据进行迭代计算
            s = trainList[i]
            sum0 = a * x[s] + b - y[s]
            sum1 = (a * x[s] + b - y[s]) * x[s]
            error1 = error1 + (a * x[s] + b - y[s])**2
        a = a - sum1 * step_size / m
        b = b - sum0 * step_size / m
        print('a=%f,b=%f' % (a, b))
        if error1 - error0 < esp:
            break
        n = n + 1
        if n > 2000:
            break
    return a, b


if __name__ == '__main__':
    x, y = data()
    a, b = MBGD(x, y)
    X = range(len(x))
    Y = [(a * i + b) for i in X]
    plt.scatter(x, y, color='red')
    plt.plot(X, Y, color='blue')
    plt.show()