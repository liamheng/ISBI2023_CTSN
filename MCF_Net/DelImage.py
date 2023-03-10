
import os
import numpy as np
import pandas as pd

# Images Labels
# 求label中的图片与文件中图片的交集
train_images_dir =  '../ADDA/pretrain_data/EyeQ'
label_train_file = '../ADDA/pretrain_data/EyeQ_label/Label_EyeQ_train.csv'


# 读取label中的图像名

df_gt = pd.read_csv(label_train_file)
img_list = df_gt["image"].tolist()
GT_list = df_gt["quality"].tolist()
img_name =[]
# 收集图片名以及质量标签
num = 0
# 删除不在label中的图像
print(img_list)
for filename in os.listdir(train_images_dir):              #listdir的参数是文件夹的路径
     # 截取图像名字进行对比
     image_name = filename[:-4]+".jpeg"
     count = img_list.count(image_name)
     img_name.append(filename)
     if count < 1 :
          del_path = image_name[:-5]+".png"
          # os.remove(os.path.join(train_images_dir,del_path))
          num=num+1
print(num)

# 删除不存在图片的标签
delrow = 0
for i in range(len(img_list)):
     name = img_list[i][:-5]+".png"
     count = img_name.count(name)
     # label是否有图片
     if count < 1 :
          img_list[i]=3
          GT_list[i]=3
          delrow = delrow+1
print(delrow)

for i in range(delrow):
     img_list.remove(3)
     GT_list.remove(3)

result={}
result['image'] = img_list
result['quality'] = GT_list
out_df = pd.DataFrame(result)
print(out_df)
# 写入csv
name_older = ['image','quality'] # 对应的列名
out_df.to_csv(label_train_file, columns=name_older)


# for filename in os.listdir(train_images_dir):
#      num = num + 1
#
# print(num)

