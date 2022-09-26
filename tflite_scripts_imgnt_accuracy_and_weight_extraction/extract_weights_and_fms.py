from inspect import currentframe
import json
from operator import mod
from sklearn import utils
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

from models_archs import utils

MODEL_NAME = 'mob_v2'
PRECISION = 8
np.random.seed(0)

tflite_models_dir = pathlib.Path("./")
tflite_model_quant_file = tflite_models_dir/(MODEL_NAME + '_' + str(PRECISION) +".tflite")

interpreter = tf.lite.Interpreter(model_path=str(tflite_model_quant_file), experimental_preserve_all_tensors=True)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

#prepare image
test_image = np.random.randint(low=32, high= 160 , size=(224, 224, 3)).astype(np.uint8)
image_batch = np.expand_dims(test_image, axis = 0)
# invoke mode
interpreter.set_tensor(input_details["index"], image_batch)
interpreter.invoke()

tensor_details = interpreter.get_tensor_details()

layers_types = utils.read_layers_types()
layers_weights = utils.read_layers_weights(layers_types)
layers_inputs = utils.read_layers_inputs()
layers_outputs = utils.read_layers_outputs()
layers_strides = utils.read_layers_strides()
expansion_projection = utils.read_expansion_projection()

assigned_layers = [0] * len(layers_weights)

def is_it_weights(tensor_shape):
    global assigned_layers
    for i in range(len(layers_weights)):
        if assigned_layers[i] == 1:
            continue
        if len(tensor_shape) == 4:
            if tensor_shape[0] == layers_weights[i].num_of_filters and tensor_shape[1] == layers_weights[i].depth \
                and tensor_shape[2] == layers_weights[i].height and tensor_shape[3] == layers_weights[i].width:
                assigned_layers[i] = 1
                return i
        if len(tensor_shape) == 3:
            if tensor_shape[0] == layers_weights[i].num_of_filters and tensor_shape[1] == layers_weights[i].height \
                and tensor_shape[2] == layers_weights[i].width:
                assigned_layers[i] = 1
                return i
        if len(tensor_shape) == 2:
            if tensor_shape[0] == layers_weights[i].num_of_filters and tensor_shape[1] == layers_weights[i].depth:
                assigned_layers[i] = 1
                return i
    
    return -1

def is_it_fms(tensor_shape):
    for i in range(len(layers_inputs)):
        if len(tensor_shape) == 3:
            if tensor_shape[0] == layers_inputs[i].depth and tensor_shape[1] == layers_inputs[i].height \
                and tensor_shape[2] == layers_inputs[i].width:
                return True
    
    return False


test_image = np.transpose(test_image, (2, 1, 0))
test_image = np.reshape(test_image, (test_image.size))
np.savetxt('fms/input_image', test_image, fmt='%i')

fms_count = 0
layer_count = 0

for t in interpreter.get_tensor_details():
    #print('*****************************')
    #print(t['index'], t['name'], interpreter.get_tensor(t['index']).shape )
    current_tensor = interpreter.get_tensor(t['index'])#.astype(np.int8)
    #if t['index'] == 3:
    #    print(t)
    current_tensor = np.squeeze(current_tensor)
    if current_tensor.ndim == 3:
        current_tensor = np.transpose(current_tensor, (2, 1, 0))
    elif current_tensor.ndim == 4:
        current_tensor = np.transpose(current_tensor, (0, 3, 2, 1))

    current_tensor_shape_str_rep = str([i for i in current_tensor.shape]).replace(' ','').replace('[', '').replace(']', '').replace(',','_')
    index_in_weights = is_it_weights(current_tensor.shape)
    if index_in_weights >= 0:
        current_tensor = np.reshape(current_tensor, (current_tensor.size))
        weights_file_sgement = str(index_in_weights) + '_' + layers_types[index_in_weights] 
        np.savetxt('./weights/weights_' + weights_file_sgement + '.txt', current_tensor, fmt='%i')
        with open('./weights/weights_' + weights_file_sgement + '_specs.txt', 'w') as f:
            f.write(str(t))
    elif is_it_fms(current_tensor.shape):
        current_tensor = np.reshape(current_tensor, (current_tensor.size))
        np.savetxt('fms/fms_' + str(fms_count) + '_' + current_tensor_shape_str_rep + '.txt', current_tensor, fmt='%i')
        with open('./fms/fms_' + str(fms_count) + '_' + current_tensor_shape_str_rep + '_specs.txt', 'w') as f:
            f.write(str(t))
        fms_count += 1
    # elif current_tensor.size < 2048:
    #     #print(current_tensor.shape)
    #     np.savetxt('./non_conv_layers/layer_' + str(layer_count) + '_' + current_tensor_shape_str_rep + '.txt', current_tensor, fmt='%i')
    #     layer_count += 1