#利用Sobel算子进行图片的边缘检测#
#-----------------------------#
import numpy as np
from PIL import Image
from scipy import signal
sobel_1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
#打开图片
img = Image.open('D:\\设计\\图片\\边缘检测图片\\1.png')
#图片转为灰度（共生矩阵为3*3）
img_grey = img.convert("L")
#转换为矩阵
I = np.asarray(img_grey)
#矩阵卷积
P = signal.convolve2d(sobel_1, I)
O = signal.convolve2d(sobel_2, I)
R = abs(P) + abs(O)
#矩阵转换为图片
R = Image.fromarray(255 - R)
R.show()
