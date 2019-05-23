# Spatial Pyramid Pooling 空間金字塔池化

via [CNN在分類圖片時圖片大小不一怎麼辦？@知呼](https://www.zhihu.com/question/45873400)

最近也遇到題主所說的問題，即傳統CNN需要輸入圖片大小固定。

1. 為什麼需要輸入圖片的大小固定

   我們知道卷積層對於圖像的大小是沒有要求的，一般的捲積核都是 $3\times3$, $5\times5$等 ，而輸入圖像一般不會小於這個大小。所以問題就是出在全連接層。

   我們假設全連接層到輸出層之間的參數是 $W^{f*o}$

   - $f$ 表示全連接層的節點個數
   - $o$ 表示輸出層的節點個數

   很顯然 $o$ 一般是固定的，而 $f$ 則會隨著輸入圖像大小的變化而變化。
   
2. 解決辦法

   1. resize or crop

      這種方法比較粗暴，而且會在預處理環節增加很大的計算量，一般而言是沒有辦法的辦法。

   2. SPP(Spatial Pyramid Pooling 空間金字塔池化)

      只需要在全連接層加上SPP layer就可以很好的解決題主的問題

      介紹參考(寫的很好懂)：

      - [深度學習筆記（一）空間金字塔池化閱讀筆記Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://blog.csdn.net/liyaohhh/article/details/50614380)
      
      - [SPP-net原理介紹](https://x-algo.cn/index.php/2017/01/13/1587/)
      
      代碼實現參考：[Pytorch實現spp net](http://www.erogol.com/spp-network-pytorch/)
      
      ```python
      import math
      from collections import OrderedDict
      import torch.nn as nn
      import torch.nn.init as init
      import torch as th
      import torch.nn.functional as F
      from torch.autograd import Variable
      
      
      class SPPLayer(nn.Module):
      
          def __init__(self, num_levels, pool_type='max_pool'):
              super(SPPLayer, self).__init__()
      
              self.num_levels = num_levels
              self.pool_type = pool_type
      
          def forward(self, x):
              bs, c, h, w = x.size()
              pooling_layers = []
              for i in range(self.num_levels):
                  kernel_size = h // (2 ** i)
                  if self.pool_type == 'max_pool':
                      tensor = F.max_pool2d(x, kernel_size=kernel_size,
                                            stride=kernel_size).view(bs, -1)
                  else:
                      tensor = F.avg_pool2d(x, kernel_size=kernel_size,
                                            stride=kernel_size).view(bs, -1)
                  pooling_layers.append(tensor)
              x = th.cat(pooling_layers, dim=-1)
              return x
      
      class DetectionNetSPP(nn.Module):
          """
          Expected input size is 64x64
          """
      
          def __init__(self, spp_level=3):
              super(DetectionNetSPP, self).__init__()
              self.spp_level = spp_level
              self.num_grids = 0
              for i in range(spp_level):
                  self.num_grids += 2**(i*2)
              print(self.num_grids)
                  
              self.conv_model = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, 128, 3)), 
                ('relu1', nn.ReLU()),
                ('pool1', nn.MaxPool2d(2)),
                ('conv2', nn.Conv2d(128, 128, 3)),
                ('relu2', nn.ReLU()),
                ('pool2', nn.MaxPool2d(2)),
                ('conv3', nn.Conv2d(128, 128, 3)), 
                ('relu3', nn.ReLU()),
                ('pool3', nn.MaxPool2d(2)),
                ('conv4', nn.Conv2d(128, 128, 3)),
                ('relu4', nn.ReLU())
              ]))
              
              self.spp_layer = SPPLayer(spp_level)
              
              self.linear_model = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(self.num_grids*128, 1024)),
                ('fc1_relu', nn.ReLU()),
                ('fc2', nn.Linear(1024, 2)),
              ]))
      
          def forward(self, x):
              x = self.conv_model(x)
              x = self.spp_layer(x)
              x = self.linear_model(x)
              return x
      ```
      
      
      
      
