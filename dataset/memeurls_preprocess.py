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
    new_size = mini - mini % 128
    mult = new_size//128
    pre_img = img[0:new_size, 0:new_size]    # 將圖片長寬裁為128的倍數
    comp_img = np.zeros((128, 128, 3))
    for i in range(128):
        for j in range(128):
            comp_img[i][j] = np.mean(
                pre_img[i*mult:(i+1)*mult, j*mult:(j+1)*mult, :3], axis=(0, 1))    # 將圖片壓縮成128*128
    comp_img = np.expand_dims(comp_img, axis=0)    # 增加第一個batch維度
    imgs.append(comp_img)    # 把圖片數組加到一個列表裡面
    if count % 10 == 0:
        clear_output(wait=True)
    count = count + 1
    print("no.%s image loaded." % count)

total = count    # 追蹤有多少有效資料(GIF數量約一成)
print("A total of %s image loaded." % total)