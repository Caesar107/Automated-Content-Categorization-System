# -*- coding: utf-8 -*-
import os
from model_file.mobilenet_arcface import MobileNet_Arcface
from tensorflow.keras.models import Model

model = MobileNet_Arcface(input_shape=(112, 96, 3), num_feature=128, classes=10572)
model.load_weights('checkpoints/model_05-6.29.h5')
print(model.input[0])
print(model.layers[-3].output)
inference_model = Model(
    inputs=model.input[0], outputs=model.layers[-3].output)
inference_model.save('model_data/mobilenet_arcface_model.h5')
