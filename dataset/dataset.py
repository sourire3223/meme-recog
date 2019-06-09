%matplotlib inline
%env KERAS_BACKEND=tensorflow
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense
from keras.datasets import cifar100
from keras.utils import np_utils
from keras.preprocessing import image
from spp.SpatialPyramidPooling import SpatialPyramidPooling    # 需引入相關檔案 https://github.com/yhenon/keras-spp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import os 
from PIL import Image
from io import BytesIO
from IPython.display import clear_output


def clear():
    os.system( 'cls' )

# 引入CIFAR，實際上只需要圖片的部分
(CIFAR_train, dum1), (dum2, dum3) = cifar100.load_data(label_mode='fine')

# 引入meme的database
imgs = []
sele_img = CIFAR_train[:8000]    # 只需要八千筆就好
for k in range(8000):
    pre_img = sele_img[k]/255
    span_img = np.zeros((128,128, 3))
    for u in range(32):
        for v in range(32):
            for i in range(4):
                for j in range(4):
                    span_img[4*u+i][4*v+j] = pre_img[u, v, :3]    # 將圖片放大成128*128
    span_img = np.expand_dims(span_img, axis=0)    # 增加第一個batch維度
    imgs.append(span_img)    # 把圖片數組加到一個列表裡面   



fr = open("memeurls.csv", 'r')
for l in fr:
    break
l = l.split(',')
count = 0
for url in l:
    res = requests.get(url)
    img0 = np.array(Image.open(BytesIO(res.content)))    # 將圖片轉為數組
    if len(img0.shape) == 2:    # 2維代表是GIF，需要排除
        continue
    img = img0/255 
    mini = min(img.shape[0], img.shape[1])
    new_size = mini - mini%128
    mult = new_size//128
    pre_img = img[0:new_size, 0:new_size]    # 將圖片長寬裁為128的倍數
    comp_img = np.zeros((128,128, 3))
    for i in range(128):
        for j in range(128):
            comp_img[i][j] = np.mean(pre_img[i*mult:(i+1)*mult, j*mult:(j+1)*mult, :3], axis=(0,1))    # 將圖片壓縮成128*128
    comp_img = np.expand_dims(comp_img, axis=0)    # 增加第一個batch維度
    imgs.append(comp_img)    # 把圖片數組加到一個列表裡面
    if count%10 == 0:
        clear_output(wait=True)
    count = count + 1
    print("no.%s image loaded."%count)
    
total = count    # 追蹤有多少有效資料(GIF數量約一成)
print("A total of %s image loaded."%total)

x2_train = np.concatenate([x for x in imgs])    # 把所有圖片數組concatenate在一起
y2_train = np.zeros((8000+total,2))
for i in range(8000):
    y2_train[i] = [1, 0]    #前八千筆的label是[1,0]，代表不是meme
for i in range(8000,8000+total):
    y2_train[i] = [0, 1]    #八千筆後的label是[0,1]，代表是meme
x1_train = x2_train
y1_train = y2_train    # 初始化x1_train跟y1_train    
count = 0
indices = range(8000+total)    #8000+total為總資料數量
indices = np.array(indices)
np.random.shuffle(indices)    #將index打亂
for i in indices:
    x1_train[count] = x2_train[i]
    y1_train[count] = y2_train[i]    #把打亂後的index依序填入新的陣列
    count = count + 1
x0_train = x1_train[:7000+total]
y0_train = y1_train[:7000+total]
x0_test = x1_train[7000+total:]
y0_test = y1_train[7000+total:]    #切出後面1000筆作為test set

#x0_train.shape
#y0_train.shape
#x0_test.shape
#y0_test.shape
