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