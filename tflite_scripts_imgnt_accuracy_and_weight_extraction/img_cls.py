import json
from statistics import mode
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow.keras.applications.mobilenet as mob_v1
import tensorflow.keras.applications.mobilenet_v2 as mob_v2
import tensorflow.keras.applications.efficientnet as eff_b0
import tensorflow.keras.applications.resnet50 as resnet
import tensorflow.keras.applications as models
from MnasNet_models.MnasNet_models import Build_MnasNet
import time
import pathlib

import sys

proxylessnas_dir = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/proxylessnas'
sys.path.append(proxylessnas_dir)

from proxyless_nas_tensorflow import proxyless_cpu

DATA_PATH = '/media/SSD2TB/shared/vedliot_evaluation/D3.3_Accuracy_Evaluation/imagenet/imagenet_val2012'

fibha_images = {}
with open('predictions_cpu.json') as json_file:
    data = json.load(json_file)
    for i in range(len(data)):
        fibha_images[data[i]['image']] = 1

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
MODEL_PATH = '/media/SSD2TB/wd/models/efficientnet_b0_no_sigmoid.h5'
PRECISION = 8

if MODEL_NAME == 'resnet50':
    model = tf.keras.models.load_model('mobilenet_v2.h5')
    print('EMBDL')
elif MODEL_NAME == 'mob_v1':
    model = models.MobileNet()
elif MODEL_NAME == 'mob_v1_0_5':
    model = models.MobileNet(alpha=0.5)
elif MODEL_NAME == 'mob_v2':
    model = models.MobileNetV2()
elif MODEL_NAME == 'mob_v2_0_5':
    model = models.MobileNetV2(alpha=0.5)
elif MODEL_NAME == 'eff_b0':
    model = models.EfficientNetB0()
elif MODEL_NAME == 'nas':
    model = models.NASNetMobile()
elif MODEL_NAME == 'mnas':
    model = Build_MnasNet('b1')
elif MODEL_NAME == 'prox':
     model = Build_MnasNet('mprox')
elif MODEL_NAME in ['eff_b0_ns_ns', 'eff_b0_no_sig', 'eff_b0_ns']:
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = models.ResNet50()

# print(model.summary())
# exit()

def representative_dataset():
    for i in range(200):
        a_test_image = load_img(test_images[i], target_size=(224, 224))
        numpy_image = img_to_array(a_test_image)
        image_batch = np.expand_dims(numpy_image, axis=0)
        if MODEL_NAME == 'mob_v1':
            processed_image = mob_v1.preprocess_input(image_batch.copy())
        elif MODEL_NAME == 'mob_v2':
            processed_image = mob_v2.preprocess_input(image_batch.copy())
        elif 'eff_b0' in MODEL_NAME:
            processed_image = eff_b0.preprocess_input(image_batch.copy())
        else:
            processed_image = resnet.preprocess_input(image_batch.copy()) 
        yield [processed_image.astype(np.float32)]

#this save is for the sake of converting to trt later by trtexec:
#first use python -m tf2onnx.convert --saved-model tensorflow-model-path --output model.onnx
#this will convert the model to onnx that can be used by trtexec but not trt scripts
#second: run trtexec and dump the output as trt engine 
#third: run the resulte using trt scripts
if PRECISION == 8:
    model.save(MODEL_NAME + "_inout")

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
limit = 1000
i=-1
processed = 0
while i < limit:
    i+= 1
    # if test_images[i].split('/')[-1] not in fibha_images:
    #     continue
    
    # processed += 1

    a_test_image = load_img(test_images[i], target_size=(224, 224))
    numpy_image = img_to_array(a_test_image)
    image_batch = np.expand_dims(numpy_image, axis=0)

    if MODEL_NAME == 'mob_v1':
        processed_image = mob_v1.preprocess_input(image_batch.copy())
    elif MODEL_NAME == 'mob_v2':
        processed_image = mob_v2.preprocess_input(image_batch.copy())
    elif 'eff_b0' in MODEL_NAME:
        processed_image = eff_b0.preprocess_input(image_batch.copy())
    else:
       processed_image = resnet.preprocess_input(image_batch.copy()) 

    if PRECISION == 32:
        t1 = time.time()
        predictions = model.predict(processed_image)[0]
        t2 = time.time()
        print('one infer per', t2 - t1, ' seconds')
    else:
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        if input_details['dtype'] == np.uint8:
            numpy_image = img_to_array(a_test_image, dtype = np.uint8)
            image_batch = np.expand_dims(numpy_image, axis = 0)
            # input_scale, input_zero_point = input_details["quantization"]
            # image_batch = image_batch / input_scale + input_zero_point
            # image_batch = image_batch.astype(np.uint8)

        interpreter.set_tensor(input_details["index"], image_batch)
        t1 = time.time()
        interpreter.invoke()
        t2 = time.time()
        print('one infer per', t2 - t1, ' seconds')
        predictions = interpreter.get_tensor(output_details["index"])[0]
        
    top5 = np.argsort(predictions)[-5:]
    top5 = np.flip(top5)
    prediction_dict = {"dets":top5.tolist(), "image": test_images[i].split('/')[-1]}
    prediction_dict_list.append(prediction_dict)

json_object = json.dumps(prediction_dict_list)

with open(MODEL_NAME + '_' + str(PRECISION) + '_' + str(limit) + "_predictions.json", "w") as outfile:
    outfile.write(json_object)