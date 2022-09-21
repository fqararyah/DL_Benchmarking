from inspect import currentframe
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

MODEL_NAME = 'mob_v2'
PRECISION = 8

tflite_models_dir = pathlib.Path("./")
tflite_model_quant_file = tflite_models_dir/(MODEL_NAME + '_' + str(PRECISION) +".tflite")

interpreter = tf.lite.Interpreter(model_path=str(tflite_model_quant_file), experimental_preserve_all_tensors=True)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

#prepare image
test_image = np.ones( (224, 224, 3), dtype=np.uint8)
image_batch = np.expand_dims(test_image, axis = 0)
# invoke mode
interpreter.set_tensor(input_details["index"], image_batch)
interpreter.invoke()

tensor_details = interpreter.get_tensor_details()

weights_count = 0
fms_count = 0
for t in interpreter.get_tensor_details():
    #print('*****************************')
    #print(t['index'], t['name'], interpreter.get_tensor(t['index']).shape )
    current_tensor = interpreter.get_tensor(t['index']).astype(np.int8)
    if t['index'] <= 10:
        current_tensor = np.squeeze(current_tensor)
        current_tensor_original_shape_str_rep = str([i for i in current_tensor.shape]).replace(',','_').replace('[', '_').replace(']', '')
        current_tensor = np.reshape(current_tensor, (current_tensor.size))
        np.savetxt('./weights/weights_' + str(weights_count) + current_tensor_original_shape_str_rep, current_tensor, fmt='%i')
        weights_count += 1

    if '224' in str(current_tensor.shape) or '112' in str(current_tensor.shape) or '56' in str(current_tensor.shape):
        current_tensor = np.squeeze(current_tensor)
        current_tensor_original_shape_str_rep = str([i for i in current_tensor.shape]).replace(',','_').replace('[', '_').replace(']', '')
        current_tensor = np.reshape(current_tensor, (current_tensor.size))
        np.savetxt('fms/fms_' + str(fms_count) + current_tensor_original_shape_str_rep, current_tensor, fmt='%i')
        fms_count += 1 