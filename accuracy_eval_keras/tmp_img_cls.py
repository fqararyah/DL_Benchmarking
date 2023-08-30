import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow.keras.applications.mobilenet_v2 as mob_v2
import tensorflow.keras.applications as models

import pathlib

DATA_PATH = '/home/fareed/wd/vedliot/D3.3_Accuracy_Evaluation/imagenet/images'


def locate_images(path):
    image_list = []
    # dirs=directories
    for (root, dirs, file) in os.walk(path):
        for f in file:
            if '.JPEG' in f:
                image_list.append(os.path.abspath(os.path.join(path, f)))
                #print(image_list[-1])
    return image_list

test_images = locate_images(DATA_PATH)

MODEL_NAME = 'mob_v1'
PRECISION = 32

model = models.MobileNetV2()

def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 224, 224, 3)
      yield [data.astype(np.float32)]

if PRECISION == 8:
    tflite_models_dir = pathlib.Path("./")
    tflite_model_quant_file = tflite_models_dir/(MODEL_NAME + '_' + str(PRECISION) +".tflite")
    if not os.path.exists(tflite_model_quant_file):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        # Ensure that if any ops can't be quantized, the converter throws an error
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Set the input and output tensors to uint8 (APIs added in r2.3)
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        tflite_model_quant = converter.convert()

        
        tflite_model_quant_file.write_bytes(tflite_model_quant)

    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
    interpreter.allocate_tensors()

prediction_dict_list = []

#limit = len(test_images)
limit = 1000
for i in range(limit):
    a_test_image = load_img(test_images[i], target_size = (224, 224))
    numpy_image = img_to_array(a_test_image)
    image_batch = np.expand_dims(numpy_image, axis = 0)

    processed_image = mob_v2.preprocess_input(image_batch.copy())

    if PRECISION == 32:
        predictions = model.predict(processed_image)[0]
    else:
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        if input_details['dtype'] == np.uint8:
            numpy_image = img_to_array(a_test_image, dtype = np.uint8)
            image_batch = np.expand_dims(numpy_image, axis = 0)
            #input_scale, input_zero_point = input_details["quantization"]
            #image_batch = image_batch / input_scale + input_zero_point
            #image_batch = image_batch.astype(np.uint8)

        interpreter.set_tensor(input_details["index"], image_batch)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details["index"])[0]
        
    top5 = np.argsort(predictions)[-5:]
    top5 = np.flip(top5)
    prediction_dict = {"dets":top5.tolist(), "image": test_images[i].split('/')[-1]}
    prediction_dict_list.append(prediction_dict)

json_object = json.dumps(prediction_dict_list)

with open(MODEL_NAME + '_' + str(PRECISION) + '_' + str(limit) + "_predictions.json", "w") as outfile:
    outfile.write(json_object)