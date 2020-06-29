# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
from PIL import Image
import random
import sys


class CNN(object):
    def __init__(self, image_height, image_width, max_captcha, char_set, model_save_dir):
        reload(sys)
        sys.setdefaultencoding('utf8')
        # 初始值
        self.image_height = image_height
        self.image_width = image_width
        self.max_captcha = max_captcha
        self.char_set = char_set
        self.char_set_len = len(char_set)
        self.model_save_dir = model_save_dir  # 模型路径
        with tf.name_scope('parameters'):
            self.w_alpha = 0.01
            self.b_alpha = 0.1
        # tf初始化占位符
        with tf.name_scope('data'):
            self.X = tf.placeholder(tf.float32, [None, self.image_height * self.image_width])  # 特征向量
            self.Y = tf.placeholder(tf.float32, [None, self.max_captcha * self.char_set_len])  # 标签
            self.keep_prob = tf.placeholder(tf.float32)  # dropout值

    @staticmethod
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

    def text2vec(self, text):
        """
        转标签为oneHot编码
        :param text: str
        :return: numpy.array
        """
        text_len = len(text)
        if text_len > self.max_captcha:
            raise ValueError('验证码最长{}个字符'.format(self.max_captcha))

        vector = np.zeros(self.max_captcha * self.char_set_len)

        for i, ch in enumerate(text):
            idx = i * self.char_set_len + self.char_set.index(ch)
            vector[idx] = 1
        return vector

    def model(self):
        x = tf.reshape(self.X, shape=[-1, self.image_height, self.image_width, 1])
        print(">>> input x: {}".format(x))

        # 卷积层1
        # 定义过滤器的权重变量，参数包括尺寸、深度和当前层节点矩阵的深度，前2个纬度[3,3]代表过滤器的尺寸为3x3，第三个纬度1表示当前层的深度，第四个纬度32表示过滤器的深度
        wc1 = tf.get_variable(name='wc1', shape=[3, 3, 1, 32], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        # 定义过滤器的偏置项变量，下一层深度32，即有32个不同的偏置项
        bc1 = tf.Variable(self.b_alpha * tf.random_normal([32]))
        ###
        # conv2d实现卷积层前向传播算法，第一个参数为当前层的节点矩阵(四维矩阵，后面三个纬度对应一个节点矩阵，第一个纬度对应一个输入batch)
        # 比如在输入层，input[0,:,:,:]表示第一张图片，input[1,:,:,:]表示第二张图片
        # strides 为各个纬度上的步长(第一维和最后一维一定是1，因为卷积层的步长只对矩阵的长和宽有效)，
        # padding为填充方式，SAME表示全0填充、VALID表示不填充
        ###
        # bias_add给每一个节点加上偏置项
        # relu完成去线性化
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='SAME'), bc1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, self.keep_prob)

        # 卷积层2
        wc2 = tf.get_variable(name='wc2', shape=[3, 3, 32, 64], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bc2 = tf.Variable(self.b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, wc2, strides=[1, 1, 1, 1], padding='SAME'), bc2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.keep_prob)

        # 卷积层3
        wc3 = tf.get_variable(name='wc3', shape=[3, 3, 64, 128], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bc3 = tf.Variable(self.b_alpha * tf.random_normal([128]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, wc3, strides=[1, 1, 1, 1], padding='SAME'), bc3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self.keep_prob)
        print(">>> convolution 3: ", conv3.shape)
        next_shape = conv3.shape[1] * conv3.shape[2] * conv3.shape[3]

        # 全连接层1
        wd1 = tf.get_variable(name='wd1', shape=[next_shape, 1024], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bd1 = tf.Variable(self.b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(conv3, [-1, wd1.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, wd1), bd1))
        dense = tf.nn.dropout(dense, self.keep_prob)

        # 全连接层2
        wout = tf.get_variable('name', shape=[1024, self.max_captcha * self.char_set_len], dtype=tf.float32,
                               initializer=tf.contrib.layers.xavier_initializer())
        bout = tf.Variable(self.b_alpha * tf.random_normal([self.max_captcha * self.char_set_len]))

        with tf.name_scope('y_prediction'):
            y_predict = tf.add(tf.matmul(dense, wout), bout)

        return y_predict
