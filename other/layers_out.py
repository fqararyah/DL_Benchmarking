from posixpath import split
import sys
import numpy as np
from numpy import imag
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.keras import datasets
import datetime
import ssl
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


ssl._create_default_https_context = ssl._create_unverified_context

pretrained_model = tf.keras.applications.MobileNet()
print("** Model architecture **")
pretrained_model.summary()

# test_images = np.random.randint(low =0, high= 256, size = [10,224, 224, 3], dtype=np.uint8)

# test_images = test_images / 255.0

# counter = 0
# #while counter < 100:
# image = test_images[counter]
# print(image[0][0])
# with open('./out/image.txt', 'w') as f:
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             for k in range(image.shape[2]):
#                 f.write(str(image[i][j][k]) + ' ')
#             f.write('\n')
#exit()
#image = np.expand_dims(image, axis=0).astype(np.float32)

image = np.empty([224, 224, 3])

count = 0
with open('./out/image.txt', 'r') as f:
    for line in f:
        splits = line.strip().split(' ')
        for k in range(len(splits)):
            image[int(count / image.shape[0])][count % image.shape[0]][k] = splits[k]  
        count += 1

image = np.expand_dims(image, axis=0).astype(np.float32)

predictions = pretrained_model.predict(x = image, batch_size = 1, verbose = 0)

intermediate_outputs = {}
layers_weights = {}
for i in range(0, len(pretrained_model.layers)):
    layer_name = pretrained_model.layers[i].name
    if len(pretrained_model.get_layer(layer_name).weights) > 0:
        layers_weights[layer_name] = pretrained_model.get_layer(layer_name).weights[0].numpy()
    intermediate_layer_model = tf.keras.Model(inputs=pretrained_model.input,
                                       outputs=pretrained_model.get_layer(layer_name).output)
    intermediate_outputs[layer_name] = (intermediate_layer_model(image).numpy())
    #print(i, "---------------------------")

#if np.sum(predictions - intermediate_outputs[-1]) != 0:
#    print("Error")
#else:
np.set_printoptions(threshold=sys.maxsize)
for layer_name, layer_output in intermediate_outputs.items():
    with open('./out/' + pretrained_model.name + '/' + layer_name + '.txt', 'w') as f:
        f.write(str(layer_output.shape) + '\n')
        f.write(str(layer_output) + '\n')
    
    if layer_name in layers_weights:
        with open('./out/' + pretrained_model.name + '/weights_' + layer_name + '.txt', 'w') as f:
            f.write(str(layers_weights[layer_name].shape) + '\n')
            f.write(str(layers_weights[layer_name]) + '\n')