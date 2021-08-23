import os
import tempfile
from threading import current_thread

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


pretrained_model = tf.keras.applications.MobileNet()
print("** Model architecture **")
pretrained_model.summary()
print(len(pretrained_model.weights))
for layer in range(0, len(pretrained_model.layers)):
    #print(pretrained_model.layers[layer].get_config())
    current_layer = pretrained_model.layers[layer]
    if '_conv' in pretrained_model.layers[layer].name.lower():
        print(current_layer.kernel_size)