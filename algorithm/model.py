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
#不太確定參數能不能直接這樣餵，paper中有提到他是對每一種size逐一去train，若是照paper的作法可能要用迴圈去跑過每一種可能的size?
#如果需要針對不同大小分開train的話，是否在預處理部分需要先用大小分組好?或是有一個(說不定可行)的做法是先把每個圖片按比例縮放至長或寬為256，這樣只要針對不是256的那一項去跑迴圈好像比較容易去遍歷所有資料。


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
