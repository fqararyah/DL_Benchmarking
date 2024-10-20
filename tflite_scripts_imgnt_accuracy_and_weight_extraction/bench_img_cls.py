import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow.keras.applications.mobilenet_v2 as mob_v2
import tensorflow.keras.applications.efficientnet as eff_b0
import tensorflow.keras.applications as models
import time
import pathlib

DATA_PATH = '/media/SSD2TB/shared/vedliot_evaluation/D3.3_Accuracy_Evaluation/imagenet/imagenet_val2012'


def locate_images(path):
    image_list = []
    # dirs=directories
    for (root, dirs, file) in os.walk(path):
        for f in file:
            if '.JPEG' in f:
                image_list.append(os.path.abspath(os.path.join(path, f)))
                # print(image_list[-1])
    return image_list


test_images = locate_images(DATA_PATH)

MODEL_NAME = 'mob_v2'
PRECISION = 8

if MODEL_NAME == 'mob_v2':
    model = models.MobileNetV2()
elif MODEL_NAME == 'eff_b0':
    model = models.EfficientNetB0()
elif MODEL_NAME == 'nas':
    model = models.NASNetMobile()


def representative_dataset():
    for i in range(100):
        a_test_image = load_img(test_images[i], target_size=(224, 224))
        numpy_image = img_to_array(a_test_image)
        image_batch = np.expand_dims(numpy_image, axis=0)
        if MODEL_NAME == 'mob_v2':
            processed_image = mob_v2.preprocess_input(image_batch.copy())
        elif MODEL_NAME == 'eff_b0':
            processed_image = eff_b0.preprocess_input(image_batch.copy())
        yield [processed_image.astype(np.float32)]


if PRECISION == 8:
    tflite_models_dir = pathlib.Path("./")
    tflite_model_quant_file = tflite_models_dir / \
        (MODEL_NAME + '_' + str(PRECISION) + ".tflite")
    if not os.path.exists(tflite_model_quant_file):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        # Ensure that if any ops can't be quantized, the converter throws an error
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Set the input and output tensors to uint8 (APIs added in r2.3)
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        if MODEL_NAME == 'eff_b0':
            converter.experimental_new_quantizer = True #//enables MLIR operator  quantization
            converter.allow_custom_ops = True

        tflite_model_quant = converter.convert()

        tflite_model_quant_file.write_bytes(tflite_model_quant)

    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
    interpreter.allocate_tensors()

prediction_dict_list = []

# limit = len(test_images)
limit = 64
image_batch = img_to_array(load_img(test_images[0], target_size=(224, 224)), dtype = np.uint8)
image_batch = np.expand_dims(image_batch, axis=0)
for i in range(1, limit):
    a_test_image = load_img(test_images[i], target_size=(224, 224))
    numpy_image = img_to_array(a_test_image, dtype = np.uint8)
    numpy_image = np.expand_dims(numpy_image, axis=0)
    image_batch = np.concatenate( (image_batch, numpy_image))

#


input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
interpreter.resize_tensor_input(input_details['index'],[limit,224,224,3])
interpreter.allocate_tensors()

print(image_batch.shape)
interpreter.set_tensor(input_details["index"], image_batch)
t1 = time.time()
interpreter.invoke()
predictions = interpreter.get_tensor(output_details["index"])[0]
t2 = time.time()
print('one infer per', (t2 - t1 ) / limit, ' seconds')