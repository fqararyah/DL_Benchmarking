
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow.keras.applications.mobilenet as mob_v1
import tensorflow.keras.applications.mobilenet_v2 as mob_v2
import tensorflow.keras.applications.efficientnet as eff_b0
import tensorflow.keras.applications.resnet50 as resnet
import tensorflow.keras.applications as models
from MnasNet_models.MnasNet_models import Build_MnasNet
import time
import pathlib

import sys

model = models.MobileNetV2()

for layer_index in range(len(model.layers)):
    print(model.layers[layer_index].name)