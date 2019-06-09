## 關於圖片規格問題

由於進行train的時候若大小不同，SPP雖支援不同大小的輸入，train的時候還是只能丟同一種大小的資料。
一種解法是先丟第一種大小train，接著再用另一種大小train下去，但我們的資料來源中，非meme的圖片規格統一(32*32)且很小，meme的圖片規格雜亂，
若依序用不同大小train的話會非常耗時，再者同一種大小只會都是meme或都不是meme，在train的時候疑似會有毀滅性的後果。

最後決定將所有圖片都拉至128*128進行訓練，訓練好的模型則可以接受各種大小的輸入。


### CIFAR100圖片處理

CIFAR的圖片可直接從keras引入(跟MINST類似)
大小皆為32x32，此處用暴力手法，直接將1個pixel當成16個pixel使用，如此一來便可成為128x128的大小。

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


### imgur上的圖片處理

imgur的圖片格式都是URL，需轉為數組，並且壓縮成128x128的大小。
壓縮方式為先裁切成長寬皆為128倍數的正方形，接著再對小正方形取平均(即pooling)，得到縮小後的照片，此方法不會讓圖片變形。
裁切時考量到meme的資訊量多半來自於上半部，於是選擇裁切掉右下部分。

```python
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
