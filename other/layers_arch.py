import os
import tempfile
from threading import current_thread
from time import process_time_ns

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

pretrained_model = tf.keras.applications.ResNet152()
print("** Model architecture **")
pretrained_model.summary()
print(len(pretrained_model.weights))
print(len(pretrained_model.layers))

for layer in range(0, len(pretrained_model.layers)):
    #print(pretrained_model.layers[layer].get_config())
    current_layer = pretrained_model.layers[layer]

    if 'conv' in current_layer.name.lower() and not '_bn' in current_layer.name.lower() and \
        not '_relu' in current_layer.name.lower() and current_layer.weights:
        print('conv: ', current_layer.name, current_layer.weights[0].shape)
    # elif 'bn' in current_layer.name:
    #     print('bn: ')
    #     print(current_layer.weights[0].numpy().shape)
    #     print(current_layer.axis)
    #     print(current_layer.momentum)
    #     print(current_layer.epsilon)
    #     print(current_layer.center)
    #     print(current_layer.scale)
    #     print(current_layer.beta_initializer)
    #     print(current_layer.gamma_initializer)
    #     print(current_layer.moving_mean_initializer)
    #     print(current_layer.moving_variance_initializer)
    #     print(current_layer.beta_regularizer)
    #     print(current_layer.gamma_regularizer)
    #     print(current_layer.beta_constraint)
    #     print(current_layer.gamma_constraint)
