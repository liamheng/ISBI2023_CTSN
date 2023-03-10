
from mydataloader import EyeQQQ_loader
import os
import numpy as np
import pandas as pd
data_root = '../Kaggle_DR_dataset/'
# Images Labels

train_images_dir = data_root + '/DRIMDB'
label_train_file = '../data/Label_DRIMDB.csv'

# 收集图片名以及质量标签
image_names = []
labels = []
for filename in os.listdir(train_images_dir):              #listdir的参数是文件夹的路径
     image_names.append(filename)
     quality = filename[7:11]
     if quality == "good" :
         labels.append(0)
     else:
         labels.append(1)
# 转成numpy类型
labels = np.array(labels)
image_names = np.array(image_names)

# 构造DataFrame格式,以便写入csv
result={}
result['image'] = image_names
result['quality'] = labels
out_df = pd.DataFrame(result)
print(out_df)
# 写入csv
name_older = ['image','quality'] # 对应的列名
out_df.to_csv(label_train_file, columns=name_older)
