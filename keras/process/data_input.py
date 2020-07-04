#!/usr/bin/env python
# encoding: utf-8
from keras.preprocessing.image import img_to_array  # 图片转为array
from keras.utils import to_categorical  # 相当于one-hot
from imutils import paths
import cv2
import numpy as np
import random
import os


def load_data(path, norm_size, class_num):
    char_set = "阿啊哀唉挨矮爱碍安岸按案暗昂袄傲奥八巴扒吧疤拔把坝爸罢霸白百柏摆败拜班般斑搬板版吧疤拔把坝爸罢霸白百柏摆败拜班般斑搬板版办半"
    # data = []  # 数据x
    # label = []  # 标签y

    data = np.zeros([128, 320*160])  # 初始化
    label = np.zeros([128, 320*160])  # 初始化

    image_paths = sorted(list(paths.list_images(path)))  # imutils模块中paths可以读取所有文件路径
    random.seed(0)  # 保证每次数据顺序一致
    random.shuffle(image_paths)  # 将所有的文件路径打乱
    for i,each_path in enumerate(image_paths):
        image = cv2.imread(each_path)  # 读取文件
        # image = cv2.resize(image, (norm_size, norm_size))  # 统一图片尺寸
        image = cv2.resize(image, (320, 160))  # 统一图片尺寸
        image = img_to_array(image)
        # data.append(image)
        # 切分文件目录，类别为文件夹整数变化，从0-61.如train文件下00014，label=14
        # maker = int(each_path.split(os.path.sep)[-2])
        # label.append(maker)

        text = each_path.split(os.path.sep)[-1].split("_")[0]
        vector = np.zeros(4)
        for i, ch in enumerate(text):
            idx = char_set.index(ch)
            vector[i] = idx

        data[i, :] = image.flatten() / 255  # 存放图片信息，flatten 转为一维
        label[i, :] = to_categorical(label, num_classes=class_num)  # 存放文字信息，生成 oneHot


    # data = np.array(data, dtype="float") / 255.0  # 归一化
    # label = np.array(label)
    # label = to_categorical(label, num_classes=class_num)  # one-hot
    return data, label


def convert2gray(img):
    """
    图片转为灰度图，如果是3通道图则计算，单通道图则直接返回
    :param img:
    :return:
    """
    if len(img.shape) > 2:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img