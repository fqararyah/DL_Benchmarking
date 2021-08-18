import os
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


pretrained_model = tf.keras.applications.ResNet50()
print("** Model architecture **")
pretrained_model.summary()
similars = {}
print(len(pretrained_model.weights))
for layer in range(0, len(pretrained_model.weights)):
    similars[layer] = {}
    layer_weights = pretrained_model.weights[layer]

    if len(layer_weights.shape) < 4:
        continue
    a = K.get_value(layer_weights)

    count = 0

    if layer_weights.shape[0] * layer_weights.shape[1] == 1:
        continue

    for i in range(0, layer_weights.shape[-1]):
        for j in range(i+1, layer_weights.shape[-1]):
            if np.average( np.abs( np.abs(a[:,:,0,j]) - np.abs(a[:,:,0,i] )) ) < \
                np.average( np.abs(a[:,:,0,i]) ) / 10:
                similars[layer][i] = j
                count += 1

    print(layer_weights.shape)
    print(count/(layer_weights.shape[-1] * layer_weights.shape[-1]))

count =0
for layer, indices in similars.items():
    for ind1, ind2 in indices.items():
        if count > 10:
            break
        count += 1
        print('++++++++++++++++++++')
        print(layer, ind1, ind2)
        print(K.get_value(pretrained_model.weights[layer])[:,:,0,ind1])
        print('and')
        print(K.get_value(pretrained_model.weights[layer])[:,:,0,ind2])
        print('--------------------')
#with tf.compat.v1.Session() as sess:  print(model_weights[0].eval())