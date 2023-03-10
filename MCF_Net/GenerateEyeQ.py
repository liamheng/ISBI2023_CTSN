
from mydataloader import EyeQQQ_loader
import os
import numpy as np
import pandas as pd
data_root = '../Kaggle_DR_dataset/'
# Images Labels

label_train_file = '../data/Label_total.csv'

df_gt = pd.read_csv(label_train_file)

img_list = df_gt["image"].tolist()
GT_QA_list = df_gt["quality"].tolist()
img_num = len(img_list)
print(img_num)
for i in range(0,img_num):
    quality = GT_QA_list[i]
    if quality == 1 : # 将需要删除的值改为3
        GT_QA_list[i] = 3
        img_list [i] = 3
    if quality == 2 :
        GT_QA_list[i] = 1

# 计算要删除的行数
delrow = img_list.count(3)

for i in range(delrow):
    GT_QA_list.remove(3)
    img_list.remove(3)




# 构造DataFrame格式,以便写入csv
result={}
result['image'] = img_list
result['quality'] = GT_QA_list
out_df = pd.DataFrame(result)
print(out_df)
# 写入csv
name_older = ['image','quality'] # 对应的列名
out_df.to_csv(label_train_file, columns=name_older)