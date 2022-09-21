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


MODEL_NAME = 'slice'
CREATE_NEW_TFLITE_MODEL_ANYWAY = True
PRECISION = 8

def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 224, 224, 3)
      yield [data.astype(np.float32)]

model = models.MobileNetV2()

cloned_model = Sequential()

for i in range(10):
    layer = model.layers[i]
    config = layer.get_config()
    weights = layer.get_weights()
    cloned_layer = type(layer).from_config(config)
    cloned_layer.build(layer.input_shape)
    cloned_layer.set_weights(weights)
    cloned_model.add(cloned_layer)


# layer = model.layers[4]
# print('#######################################', layer.name)
# config = layer.get_config()
# weights = layer.get_weights()
# cloned_layer = type(layer).from_config(config)
# cloned_layer.build(layer.input_shape)
# cloned_layer.set_weights(weights)
# cloned_model.add(cloned_layer)

print(cloned_model.summary)

tflite_models_dir = pathlib.Path("./")
tflite_model_quant_file = tflite_models_dir/(MODEL_NAME + '_' + str(PRECISION) +".tflite")

if CREATE_NEW_TFLITE_MODEL_ANYWAY or not os.path.exists(tflite_model_quant_file):
    converter = tf.lite.TFLiteConverter.from_keras_model(cloned_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model_quant = converter.convert()

    
    tflite_model_quant_file.write_bytes(tflite_model_quant)

interpreter = tf.lite.Interpreter(model_path=str(tflite_model_quant_file), experimental_preserve_all_tensors=True)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]

output_details = interpreter.get_output_details()[0]
test_image = np.ones( (224, 224, 3), dtype=np.uint8)
image_batch = np.expand_dims(test_image, axis = 0)

interpreter.set_tensor(input_details["index"], image_batch)
interpreter.invoke()
predictions = interpreter.get_tensor(output_details["index"])[0]

print(predictions.shape)