#分别用最小二乘法和Sklearn包实现线性回归
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
A = 2
B = 3
#生成两个100行1列的矩阵
xArr = np.random.randint(100, size=(100, 1))
eArr = np.random.randint(200, size=(100, 1))
#定义yArr
yArr = A + B * xArr + eArr
#保存为Excel文件，分别保存为X_data和Y_data列,并从excel文件读取数据
data = np.c_[xArr, yArr]
save = pd.DataFrame(data, columns=["x_data", "y_data"])
save.to_excel("Out.xls", index=False)  #index = False
out = pd.read_excel("Out.xls")
print(out)
#最小二乘法
x_b = np.c_[xArr, np.ones((100, 1), int)]
w_best = (np.linalg.inv(x_b.T.dot(x_b))).dot(x_b.T).dot(yArr)
y_result = np.dot(x_b, w_best)  #x_b
#sklearn
regr = LinearRegression().fit(xArr, yArr)
#解决标题中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#绘图
plt.figure()
#-----------------
plt.subplot(1, 2, 1)
plt.title('最小二乘法')
plt.scatter(xArr, yArr)
plt.plot(xArr, y_result, 'r')
#-----------------
plt.subplot(1, 2, 2)
plt.title('Sklearn')
plt.scatter(xArr, yArr)
plt.plot(xArr, regr.predict(xArr), 'g')
#-----------------
plt.show()
