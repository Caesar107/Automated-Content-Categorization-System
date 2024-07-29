import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# from model_mobilefacenet.mobilefacenet import *
# from model_mobilefacenet.mobilefacenet_func import *
# from model_mobilefacenet.arcface_func import *

# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

# To find out which devices your operations and tensors are assigned to
tf.debugging.set_log_device_placement(True)

cls_num = 10572

def preprocess(img):
    img = (img.astype('float32') - 127.5) / 128.0
    img = np.expand_dims(img, axis=0)
    return img


# load image
img_yzy = preprocess(plt.imread("data/test/yzy/yzy.jpg"))
img_lm = preprocess(plt.imread("data/test/lm/lm.jpg"))
img_ly = preprocess(plt.imread("data/test/ly/d03a2949fe4d6aa5e095e5945d84593.jpg"))
img_zt = preprocess(plt.imread("data/test/zt/zt.jpg"))

img_test = preprocess(plt.imread("data/test0.jpg"))


if __name__ == '__main__':
    print("111111111")
    # feed forward
    model = tf.keras.models.load_model("model_data/mobilefacenet_model.h5")

    embedding_yzy = model.predict(img_yzy)
    embedding_lm = model.predict(img_lm)
    embedding_zt = model.predict(img_zt)
    embedding_ly = model.predict(img_ly)

    embedding_test = model.predict(img_test)

    # test result 计算余弦距离
    embedding_yzy = embedding_yzy / np.expand_dims(np.sqrt(np.sum(np.power(embedding_yzy, 2), 1)), 1)
    embedding_lm = embedding_lm / np.expand_dims(np.sqrt(np.sum(np.power(embedding_lm, 2), 1)), 1)
    embedding_zt = embedding_zt / np.expand_dims(np.sqrt(np.sum(np.power(embedding_zt, 2), 1)), 1)
    embedding_ly = embedding_zt / np.expand_dims(np.sqrt(np.sum(np.power(embedding_ly, 2), 1)), 1)
    embedding_test = embedding_test / np.expand_dims(np.sqrt(np.sum(np.power(embedding_test, 2), 1)), 1)

    # get result
    print(np.sum(np.multiply(embedding_yzy, embedding_test), 1))
    print(np.sum(np.multiply(embedding_lm, embedding_test), 1))
    print(np.sum(np.multiply(embedding_zt, embedding_test), 1))
    print(np.sum(np.multiply(embedding_ly, embedding_test), 1))
    print("over")
    # # save database
    # db = np.concatenate((embedding_yzy, embedding_lm, embedding_steve), axis=0)
    # print(db.shape)
    # np.save("pretrained_model/db", db)
