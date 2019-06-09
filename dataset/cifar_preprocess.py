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