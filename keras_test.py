# CNN模型训练

# -*- coding: utf-8 -*-
import cv2
import os
import random
import numpy as np
from keras.models import *
from keras.layers import *
from keras import callbacks

import os

char_lst = []
for file in os.listdir('sample/origin'):
    chars = file.replace('.png', '')
    # if len(chars) != 2:
    # print(file)
    for char in chars[:2]:
        char_lst.append(char)

# print(char_lst)
characters = ''.join(list(set(char_lst)))
width, height, n_len, n_class = 320, 160, 4, len(characters)
# print(n_class)

print(characters)

# 产生训练的一批图片，默认是32张图片
def gen(dir, batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for _ in range(n_len)]
    files = os.listdir(dir)
    while True:
        for i in range(batch_size):
            path = random.choice(files)
            imagePixel = cv2.imread(dir + '/' + path, 1)
            # filename = path[:2]
            filename = path.split("_")[0]
            # print(filename)
            X[i] = imagePixel
            for j, ch in enumerate(filename):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1

        yield X, y


input_tensor = Input((height, width, 3))
x = input_tensor

# 产生有四个block的卷积神经网络
for i in range(4):
    # 卷积层
    x = Conv2D(32 * 2 ** i, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32 * 2 ** i, (3, 3), activation='relu', padding='same')(x)
    # 池化层
    x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.25)(x)

# 多输出模型，使用了4个'softmax'來分别预测2個汉字的输出
x = [Dense(n_class, activation='softmax', name='c%d' % (i + 1))(x) for i in range(4)]
model = Model(inputs=input_tensor, outputs=x)
model.summary()

# 保存模型结构图
# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file="./model.png", show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# 保存效果最好的模型
cbks = [callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)]

train_dir = 'sample/origin'
test_dir = 'sample/origin'
history = model.fit_generator(gen(train_dir, batch_size=8),  # 每次生成器会产生8张小批量的图片
                              steps_per_epoch=int(len(char_lst) / 16),  # 每次的epoch要训练120批图片
                              epochs=10000,  # 总共训练次
                              callbacks=cbks,  # 保存最好的模型
                              validation_data=gen(test_dir),  # 验证数据也是用生成器來产生
                              validation_steps=10  # 用10组图片来进行验证
                              )

# 绘制损失值图像
import matplotlib.pyplot as plt


def plot_train_history(history, train_metrics, val_metrics):
    plt.plot(history.history.get(train_metrics), '-o')
    plt.plot(history.history.get(val_metrics), '-o')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])


# 打印整体的loss与val_loss，并保存图片
plot_train_history(history, 'loss', 'val_loss')
plt.savefig('./all_loss.png')

plt.figure(figsize=(12, 4))

# 第一个数字的正确率
plt.subplot(2, 2, 1)
plot_train_history(history, 'c1_accuracy', 'val_c1_accuracy')

# 第二个数字的正确率
plt.subplot(2, 2, 2)
plot_train_history(history, 'c2_accuracy', 'val_c2_accuracy')

# 第三個数字的正確率
plt.subplot(2, 2, 3)
plot_train_history(history, 'c3_accuracy', 'val_c3_accuracy')

# 第四個数字的正确率
plt.subplot(2, 2, 4)
plot_train_history(history, 'c4_accuracy', 'val_c4_accuracy')

# 保存图片
plt.savefig('./train.png')
plt.show()