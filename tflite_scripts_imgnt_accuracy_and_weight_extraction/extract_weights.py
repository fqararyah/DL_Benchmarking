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


for t in interpreter.get_tensor_details():
    print('*****************************')
    print(t['index'], t['name'], interpreter.get_tensor(t['index']).shape )

# for dict in tensor_details:
#     i = dict['index']
#     tensor_name = dict['name']
#     scales = dict['quantization_parameters']['scales']
#     zero_points = dict['quantization_parameters']['zero_points']
#     tensor = interpreter.tensor(i)()
    
#     if i < 50:
#         print('###################',i,tensor_name,'#####################')
#         # print('***************************************************************')
#         # print('Scaling', scales)
#         # print('***************************************************************')
#         # print('Zeros', zero_points)
#         # print('***************************************************************')
#         print('tensor', tensor.shape)
