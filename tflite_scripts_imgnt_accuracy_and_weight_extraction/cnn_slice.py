import json
from operator import mod
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow.keras.applications.mobilenet_v2 as mob_v2
import tensorflow.keras.applications as models

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

import pathlib


MODEL_NAME = 'mob_v1_slice_0_10_0_25'
CREATE_NEW_TFLITE_MODEL_ANYWAY = True
PRECISION = 32

def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 224, 224, 3)
      yield [data.astype(np.float32)]

model = models.MobileNet(alpha=0.25)

cloned_model = Sequential()

for i in range(1000):
    layer = model.layers[i]
    if len(layer.input_shape) == 4 and layer.input_shape[1] < 56:
        break
    print(layer.input_shape)
    config = layer.get_config()
    weights = layer.get_weights()
    cloned_layer = type(layer).from_config(config)
    cloned_layer.build(layer.input_shape)
    cloned_layer.set_weights(weights)
    cloned_model.add(cloned_layer)

print(cloned_model.summary())

cloned_model.save(MODEL_NAME + "_inout")