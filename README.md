# meme-recog
迷因圖片辨識 Meme Recognition Project

## 專案概要資訊

- 組員：[林琮珉](https://www.facebook.com/100000215725013)(組長)、[林建銘](https://www.facebook.com/100002583373383)、[蕭維斌](https://www.facebook.com/weipin.hsiao.1)(專案管理)

- 主題：迷因圖片辨識 Meme Recognition Project

- 介紹：

  ![moneyyyy](README.assets/Ssns07U.jpg)

  Meme is life, meme is love!

  迷因(meme)是這世界上最重要的東西了，迷因的媒介非常多，語句、圖片、影片都可以形成迷因，迷因一但產生便會病毒性地擴散，產生大量的變體，因此迷因通常具備可塑性高與極為泛用等特徵。

  本組期末報告希望藉由蒐集迷因圖片與非迷因圖片，利用神經網路訓練一個可以辨識圖片是否為迷因圖的人工智慧模型。

- 實作方法：

  1. 資料蒐集：整合兩個圖片資料庫：

     - Meme Generator Data Set: <https://www.kaggle.com/electron0zero/memegenerator-dataset/home>
     - cifar-100: <https://www.cs.toronto.edu/~kriz/cifar.html>

     前者作為迷因圖片，後者會被標籤為非迷因圖片

  2. 資料前處理：處理資料形式供模型使用

     主要的工作集中在 Meme Generator Data Set 的處理，其原始資料為 webarchive.loc.gov 的庫存引導網址，需要建立爬蟲把正確圖片連結下載回來，再利用 Keras API 預處裡圖片檔。

  3. 使用神經網路模型訓練之，並驗證其成功率。

  目前模型設計：

     ```mermaid
  graph LR
     Pic --> CNN
     CNN --> SPP
     SPP --> FullConectedNN
     FullConectedNN --> IsMeme?
     ```

- 專案排程：

  5/27(專案目前進度) 開始處裡 Meme Generator Data Set，完成爬蟲，開始爬資料，並嘗試設計預處理程式碼

  5/28 資料預處裡程式碼工作完成，開始執行預處裡工作

  6/1 完成模型設計並利用小規模資料測試，先期成果發表

  6/2-6/6 實際訓練模型並分析結果

  6/7 繳交報告並結案

---

# 專案報告

[TOC]

## 研究問題

蒐集迷因圖片與非迷因圖片，利用神經網路訓練一個可以辨識圖片是否為迷因圖的人工智慧模型。

## 資料蒐集與預處理

研究蒐集的資料來主要來自兩個來源，一個為作為參照用的非迷因圖片集，一個為迷因圖片集。

### 迷因圖片來源 - 1

本研究使用 kaggle 上的 [Meme Generator Data Set](https://www.kaggle.com/electron0zero/memegenerator-dataset/home)，其內容為 57,652 張不同之迷因圖片，在美國國會圖書館(Library of Congress)設置之網頁庫存([Library of Congress Web Archives](http://webarchive.loc.gov/))之連結，所有的迷因圖片皆源自於 [Meme Generator](memegenerator.net) 此一迷因圖片網站，此網站提供網友自製迷因圖片之工具，同時也接受迷因圖片投稿，是一重要迷因集散地。

#### 資料格式

該資料集檔案格式為 csv，其內容樣本如下：

| Meme ID | Archived URL | Base Meme Name | Meme Page URL | MD5 Hash | File Size (In Bytes) | Alternate Text |
| --- | --- | --- | --- | --- | --- | --- |
| 10509464 | http://webarchive.loc.gov/all/20150508184553/http://cdn.meme.am/instances/10509464.jpg | Spiderman Approves | http://memegenerator.net/instance/10509464 | 5be4b65cc32d3a57be5b6693bb519155 | 24093 | seems legit |

#### 連結處理

由於 Archived URL 並無法直接下載到圖片，同時連結上的圖片也需要先行處理才能供 Keras 使用，首先我們先撰寫下列程式碼將利用 `BeautifulSoup` 將連結轉換成可直接下載圖片之連結：

```python
import threading
import numpy as np
import requests
from bs4 import BeautifulSoup, SoupStrainer

num_thread = 240
_in_file = ["" for i in range(num_thread)]
# 子執行緒類別
class MyThread(threading.Thread):
    def __init__(self, num):
        threading.Thread.__init__(self)
        self.num = num 

    def run(self):
        self.upp = (self.num + 1) * (57652 // num_thread + 1)
        self.low = self.num * (57652 // num_thread + 1)
        fr = open("memegenerator.csv", 'r', encoding="utf-8")
        self.count = 0
        for l in fr:
            break
        for l in fr:
            self.count += 1
            if self.count <= self.low or self.count > self.upp:
                continue

            l = l.split(',')

            url = l[1]
            html = requests.get(url)
            soup = BeautifulSoup(html.content, "html.parser")
            sel = soup.select(".day a")
            l[1] = url[-12:] + " not found"
            
            for i in sel:
                if not i.has_attr('href'):
                    continue
                if requests.get(i['href']):
                    l[1] = i['href']
                    break

        #     print(",".join(l))
            _in_file[self.num] += (",".join(l))

# 建立 num_thread 個子執行緒
threads = []
for i in range(num_thread):
    threads.append(MyThread(i))
    threads[i].start()

# 主執行緒繼續執行自己的工作
# ...

# 等待所有子執行緒結束
for i in range(num_thread):
    threads[i].join()

fw = open("memegenerator_image_url_new.csv", 'w', encoding="utf-8")
fw.write("".join(_in_file))
fw.close()
print("Done.")
```

### 迷因圖片來源 - 2

由於專案執行期間內遭愈美國國會圖書館設置之網頁庫存當機，我們將專案使用的圖片改至搜尋 imgur 之 meme 圖片，由爬蟲自動抓取帶有 #meme tag 之圖片連結再由我們檢視圖片狀況，共收集 10,158 筆連結，爬蟲程式碼來源為組員過去作業，請參考以下連結：https://nbviewer.jupyter.org/github/sourire3223/aimath/blob/master/unrelated/B04201020.ipynb。

由於 imgur 的圖片連結，需轉為數組，並且壓縮成 128x128 的大小。壓縮方式為先裁切成長寬皆為 128 倍數的正方形，接著再對小正方形取平均(即pooling)，得到縮小後的照片，此方法不會讓圖片變形。
裁切時考量到迷因的資訊量多半來自於上半部，於是選擇裁切掉右下部分。處理完成後圖片詳見 https://drive.google.com/drive/folders/1JBKVTX9fKiHSLLjLsNE5Wlv5WJAKGwt9。

程式碼如下

```python
import numpy as np
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
```

### 非迷因圖片來源 - CIFAR100 小圖像分類數據集

該資料集 Keras 即有內建(跟MINST類似)，有 50000 張 32x32 彩色訓練圖像數據，以及 10000 張測試圖像數據，總共分為 100 個類別，直接將 1 個 pixe l當成 16 個 pixel 使用，如此一來便可成為 128x128 的大小。

```python
from keras.datasets import cifar100
(CIFAR_train, dum1), (dum2, dum3) = cifar100.load_data(label_mode='fine')    # 引入CIFAR，實際上只需要圖片的部分
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
```

### 關於圖片規格處理

由於進行 train 的時候若大小不同，SPP(Spatial Pyramid Pooling，空間金字塔池化) 雖支援不同大小的輸入，train 的時候還是只能丟同一種大小的資料。

一種解法是先丟第一種大小 train，接著再用另一種大小 train 下去，但我們的資料來源中，非迷因的圖片規格統一 (32x32) 且很小，meme 的圖片規格雜亂，若依序用不同大小 train 的話會非常耗時，再者同一種大小只會都是迷因或都不是迷因，在 train 的時候疑似會有毀滅性的後果。最後決定將所有圖片都拉至128*128進行訓練，訓練好的模型則可以接受各種大小的輸入。

本報告在進入模型前資料甚於資料整合程式碼如下：

```python
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
```

本章節相關程式碼詳見本專案 dataset 資料夾。

## 模型

### 理論說明 - 空間金字塔池化(Spatial Pyramid Pooling)

本報告由於可能會需要處理不同大小之圖片，利用改變大小或剪裁可能力有未逮，故在此考量下我們引用了金字塔空間池化，其詳細之論文可參考：[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/abs/1406.4729)

金字塔空間池化，簡稱 SPP，當處設計就是為了處理不同大小之圖片。雖然 CNN 本身 convolution -> pooling -> ... -> convolution -> pooling 如此之架構，其實並不受限於圖片大小，所有圖片都能用此架構訓練，問題出在當最後一次 pooling 需要進到 fully-connected layer 時，fully-connected layer 需要確立輸入大小，而不同尺寸之圖片，在最後一次 pooling 完後，其資料矩陣大小並不一致，這就導致了處理到一半的資料無法供 fully-connected layer 訓練的問題。

![](README.assets/003.png)

傳統的處理方法，是在圖片進入 CNN 模型之前，先裁切或縮放變成一致之大小，以確保資料進入 fully-connected layer 之前大小一致。而金字塔空間池化的思維，是引入一個無論何種尺度的資料最後都會被 pooling 成一致大小之輸出，以解決進入fully-connected layer 時大小不一的問題。

![](/README.assets/004.png)

實際 SPP 處理概念是，pooling 原本是作為取得特徵值使用的，那就將原本的固定多少尺寸取一特徵值，變成固定取一定量的特徵值，而依輸入矩陣的大小修正 pooling 時的矩陣大小，將不同尺寸輸入分割成固定數量的區域，用來取特徵值。

但對大小不同的圖片，其適合取特徵值尺度也不同，SPP 採取一個簡單暴力的做法，就是引入不同規模的框架，例如上圖，就是設定取 4x4、2x2、1x1 個特徵值，如此取得固定長度的輸出，再交由 fully-connected layer 進行後續處理。

而在 Keras 中實現 SPP 的方法，可以藉由自 https://github.com/yhenon/keras-spp 下載專案壓縮檔，並在 python 環境下於解壓縮後之資料夾執行 `python setup.py install` 之指令，或於 shell 下執行：

```shell
git clone https://github.com/yhenon/keras-spp.git
sudo python setup.py install
```

### 模型設定

模型設定圖如下：此圖為模型設計，共有五層 convolution layer，其中第三層為避免縮得太小，沒有進行 pooling，而第五層由於利用 SPP 會自動壓平，就不需另行設計 flatten layer。

![model_plot](README.assets/model_plot.png)

程式碼：

```python
#資料預處理完後，應用SPP實作CNN
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adadelta
from keras import backend as K
from spp.SpatialPyramidPooling import SpatialPyramidPooling #SPP

#資料已處理完，x_train與x_test皆為( 資料筆數 , 長 , 寬 , 3(RGB) )
#y_train與y_test則為catergorical的0與1

#CNN的模型設為五層，第五層最後的pooling改用SPPnet

num_channels = 3  #RGB
num_classes = 2   #是或不是meme

model = Sequential()
model.add(Conv2D(4, (8,8), padding='same', input_shape=(None, None, 3)))   #圖片大小不固定所以用NONE
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))   #第一層
model.add(Conv2D(8, (7,7), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))   #第二層
model.add(Conv2D(16, (6,6), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))   #第三層
model.add(Conv2D(32, (5,5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))   #第四層
model.add(Conv2D(64, (4,4), padding='same'))
model.add(Activation('relu'))
#這裡運用SPP讓結果的大小是固定的
model.add(SpatialPyramidPooling([1, 2, 4])) #會輸出(1+4+16)=21的大小
model.add(Dense(num_classes))   #搭配上行是一個21*2的FC layer
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#training
model.fit(x_train, y_train, batch_size=128, epochs=10)
result = model.predict_classes(x_test)

#抽9張圖片出來看看結果
pick = np.random.randint(1,9999, 9)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x0_test[pick[i]], cmap='Greys')
    plt.title(result[pick[i]])
    plt.axis("off")
    
score = model.evaluate(x_test, y_test)    
loss, acc = score    
print(acc)    
```

### 分析結果

```
> x_train = x0_train
> y_train = y0_train
> model.fit(x_train, y_train, batch_size=500, epochs=15)    #batch_size跟epochs可以再調整

Epoch 1/10
200/200 [==============================] - 10s 51ms/step - loss: 0.5375 - acc: 0.7500
Epoch 2/10
200/200 [==============================] - 7s 35ms/step - loss: 0.5186 - acc: 0.7500
Epoch 3/10
200/200 [==============================] - 7s 35ms/step - loss: 0.5125 - acc: 0.7500
Epoch 4/10
200/200 [==============================] - 7s 35ms/step - loss: 0.5253 - acc: 0.7500
Epoch 5/10
200/200 [==============================] - 7s 35ms/step - loss: 0.4982 - acc: 0.7500
Epoch 6/10
200/200 [==============================] - 7s 35ms/step - loss: 0.4816 - acc: 0.7500
Epoch 7/10
200/200 [==============================] - 7s 35ms/step - loss: 0.4806 - acc: 0.7500
Epoch 8/10
200/200 [==============================] - 7s 35ms/step - loss: 0.4579 - acc: 0.7500
Epoch 9/10
200/200 [==============================] - 7s 37ms/step - loss: 0.4774 - acc: 0.7500
Epoch 10/10
200/200 [==============================] - 8s 39ms/step - loss: 0.4333 - acc: 0.7500

> x_test = x0_test
> y_test = y0_test
> score = model.evaluate(x_test, y_test)
> print(f'測試資料的 loss: {score[0]:.5f}')
> print(f'測試資料的正確率: {score[1]}')

測試資料的 loss: 0.36642
測試資料的正確率: 0.8205128205128205
```

正確率僅八成多(執行之程式碼放置於 [Training.ipynb](Training.ipynb))

## 討論

依 [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/abs/1406.4729) 的敘述，理論上 SPP 取得特徵值的效果會比傳統裁切縮放好很多，但由於我們尚未找到撰寫 free size 之資料形式，無法以圖案原始尺吋進行分析，可能是這個原因沒有發揮 SPP 的功能，最後僅得到約八成之結果。

感謝老師助教這半年來的課程，讓我們這些機器學習跟寫程式的初學者學到很多，謝謝。



