# import onnx
# import common

# from onnx_tf.backend import prepare

# onnx_model = onnx.load(common.MODEL_PATH)  # load onnx model
# output = prepare(onnx_model).run(input)  # run the loaded model

# test_images = common.locate_images(common.DATASET_PATH)

import common
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import json

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

pretrained_model = tf.keras.applications.resnet50.ResNet50()

test_images = common.locate_images(common.DATASET_PATH)

prediction_dict_list = []
for i in range(len(test_images)):
    test_image = image.load_img(test_images[i], target_size=(224, 224))
    x = image.img_to_array(test_image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = pretrained_model.predict(x)
    indices = np.argsort(preds[0])[-5:]
    indices = np.flip(indices)
    prediction_dict = {"dets":indices.tolist(), "image": test_images[i].split('/')[-1]}
    prediction_dict_list.append(prediction_dict)
    #print(test_images[i].split('/')[-1], indices)

json_object = json.dumps(prediction_dict_list)
#print('bad images:',bad_count)
with open("keras_predictions.json", "w") as outfile:
    outfile.write(json_object)