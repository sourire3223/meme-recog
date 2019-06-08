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

## 研究問題

蒐集迷因圖片與非迷因圖片，利用神經網路訓練一個可以辨識圖片是否為迷因圖的人工智慧模型。

## 資料蒐集

研究蒐集的資料來主要來自兩個來源，一個為作為參照用的非迷因圖片集，一個為迷因圖片集。

### 迷因圖片來源

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



### 非迷因圖片來源 - CIFAR100 小圖像分類數據集

該資料集 Keras 即有內建，有 50000 張 32x32 彩色訓練圖像數據，以及 10000 張測試圖像數據，總共分為 100 個類別，其載入程式碼如下

```python
from keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
```

### 資料前處理



## 模型



### 分析結果



## 結論





