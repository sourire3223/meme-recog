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