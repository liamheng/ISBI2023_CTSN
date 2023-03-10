# encoding: utf-8
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageCms
import os
from sklearn import preprocessing
import pandas as pd

#--------------------------------------------------
def load_eyeQ_excel(data_dir, list_file, n_class=2):
    image_names = []
    labels = []
    lb = preprocessing.LabelBinarizer()
    lb.fit(np.array(range(n_class)))
    df_tmp = pd.read_csv(list_file)
    img_num = len(df_tmp)

    # for idx in range(img_num):
    #     image_name = df_tmp["image"][idx]
    #     image_names.append(os.path.join(data_dir, image_name[:-4] + '.jpg'))
    #
    #     label = lb.transform([int(df_tmp["quality"][idx])])
    #
    #     labels.append(label)
    # 改写，one-hot二分类编码问题
    for idx in range(img_num):
        image_name = df_tmp["image"][idx]
        image_names.append(os.path.join(data_dir, image_name[:-5] + '.png'))

        label = df_tmp["quality"][idx]
        if label == 0 :
            label = np.array([1,0])
        else:
            label = np.array([0, 1])

        labels.append(label)

    return image_names, labels
def load_DRIMDB_excel(data_dir, list_file, n_class=2):
    image_names = []
    labels = []
    lb = preprocessing.LabelBinarizer()
    lb.fit(np.array(range(n_class)))
    df_tmp = pd.read_csv(list_file)
    img_num = len(df_tmp)

    for idx in range(img_num):
        image_name = df_tmp["image"][idx]
        image_names.append(os.path.join(data_dir, image_name[:-4] + '.jpg'))
        label = df_tmp["quality"][idx]
        if label == 0 :
            label = np.array([1,0])
        else:
            label = np.array([0, 1])

        labels.append(label)

    return image_names, labels
def load_FIVE_excel(data_dir, list_file, n_class=2):
    image_names = []
    labels = []
    lb = preprocessing.LabelBinarizer()
    lb.fit(np.array(range(n_class)))
    df_tmp = pd.read_csv(list_file)
    img_num = len(df_tmp)

    for idx in range(img_num):
        image_name = df_tmp["image"][idx]
        image_names.append(os.path.join(data_dir, image_name))
        label = df_tmp["quality"][idx]
        if label == 0 :
            label = np.array([1,0])
        else:
            label = np.array([0, 1])

        labels.append(label)

    return image_names, labels
class DatasetGenerator1(Dataset):
    def __init__(self, data_dir, list_file, dataset,transform1=None, transform2=None, n_class=2, set_name='train'):

        if dataset == "EyeQ":
            image_names, labels = load_eyeQ_excel(data_dir, list_file, n_class=2)
        if dataset == "DRIMDB":
            image_names, labels = load_DRIMDB_excel(data_dir, list_file, n_class=2)
        if dataset == "FIVE":
            image_names, labels = load_FIVE_excel(data_dir, list_file, n_class=2)



        self.image_names = image_names
        self.labels = labels
        self.n_class = n_class
        self.transform1 = transform1
        self.transform2 = transform2
        self.set_name = set_name

        srgb_profile = ImageCms.createProfile("sRGB")
        lab_profile = ImageCms.createProfile("LAB")
        self.rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')

        if self.transform1 is not None:
            image = self.transform1(image)

        img_hsv = image.convert("HSV")
        img_lab = ImageCms.applyTransform(image, self.rgb2lab_transform)

        img_rgb = np.asarray(image).astype('float32')
        img_hsv = np.asarray(img_hsv).astype('float32')
        img_lab = np.asarray(img_lab).astype('float32')

        if self.transform2 is not None:
            img_rgb = self.transform2(img_rgb)
            img_hsv = self.transform2(img_hsv)
            img_lab = self.transform2(img_lab)

        if self.set_name == 'train':
            label = self.labels[index]

            return torch.FloatTensor(img_rgb), torch.FloatTensor(img_hsv), torch.FloatTensor(img_lab), torch.FloatTensor(label)
        else:
            return torch.FloatTensor(img_rgb), torch.FloatTensor(img_hsv), torch.FloatTensor(img_lab)

    def __len__(self):
        return len(self.image_names)

