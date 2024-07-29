# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD,Adam
import numpy as np
from sklearn.model_selection import train_test_split

from model_mobilenet.mobilenetv2_arcface import *

# 数据路径
data_root = "data/CASIA"
img_txt_dir = os.path.join(data_root, 'CASIA-WebFace-112X96.txt')


def load_dataset(val_split=0.05):
    image_list = []     # image directory
    label_list = []     # label
    with open(img_txt_dir) as f:
        img_label_list = f.read().splitlines()
    for info in img_label_list:
        image_dir, label_name = info.split(' ')
        image_list.append(os.path.join(data_root, 'CASIA-WebFace-112X96', image_dir))
        label_list.append(int(label_name))

    trainX, testX, trainy, testy = train_test_split(image_list, label_list, test_size=val_split)

    return trainX, testX, trainy, testy


def preprocess(x,y):
    # x: directory，y：label
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3) # RGBA
    x = tf.image.resize(x, [112, 96])

    x = tf.image.random_flip_left_right(x)

    # x: [0,255]=> -1~1
    x = (tf.cast(x, dtype=tf.float32) - 127.5) / 128.0
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=class_num)

    return (x, y), y

# get data slices
train_image, val_image, train_label, val_lable = load_dataset()

# get class number
class_num = len(np.unique(train_label))

batchsize = 64
db_train = tf.data.Dataset.from_tensor_slices((train_image, train_label))     # construct train dataset
db_train = db_train.shuffle(1000).map(preprocess).batch(batchsize)
db_val = tf.data.Dataset.from_tensor_slices((val_image, val_lable))
db_val = db_val.shuffle(1000).map(preprocess).batch(batchsize)

def mobilefacenet_train():
    model = Mobilefacenet_Arcface(input_shape=(112, 96, 3), num_feature=128, classes=class_num)
    model.load_weights('pre_weight/mobilefacenet_model.h5', skip_mismatch=True, by_name=True)
    # # 优化器
    # optimizer = Adam(lr=0.001, epsilon=1e-8)
    optimizer = SGD(lr=0.1, momentum=0.9, nesterov=True)
    # 模型损失
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # 回调函数
    callback_list = [ModelCheckpoint("checkpoints/mobilenet_v2/ep{epoch:02d}-accuracy{accuracy:.3f}-loss{loss:.3f}.h5",
                                     monitor='val_loss',save_weights_only=True,
                                     verbose=1, save_best_only=False, period=2),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1),
                     TensorBoard(log_dir='logs/mobilenet_v2')]

    # 模型训练
    model.fit(db_train,
              validation_data=db_val,
              validation_freq=1,
              epochs=40, callbacks=callback_list,
              initial_epoch=0)

    # 待完成，lwf数据集上验证
    # 模型保存[输出倒数第三层数据，人脸特征]
    inference_model = Model(inputs=model.input, outputs=model.layers[-3].output)
    inference_model.save('model_data/mobilenet_v2/mobilefacenet_model.h5')
    return model


if __name__ == '__main__':
    mobilefacenet_train()

