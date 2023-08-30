
from lib2to3.pgen2.token import NAME
import os

# This sample uses an ONNX ResNet50 Model to create a TensorRT Inference Engine
import random
import sys

import numpy as np

# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit
import tensorrt as trt
from PIL import Image


import json


from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess_input


test_image = ('/home/fareed/wd/vedliot/D3.3_Accuracy_Evaluation/coco/coco_val2017/000000000885.jpg')

_test_image = image.load_img(test_image, target_size=(224, 224))
c, h, w = 3,224,224
resized_image = _test_image.resize((w, h), Image.ANTIALIAS)
np_array = np.asarray(resized_image)
if(np_array.ndim == 2):
    np_array = np.repeat(np_array[:, :, np.newaxis], 3, axis=2)
image_arr = (
    np_array
    .astype(trt.nptype(trt.float32))
)
# This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
image_arr = image_arr[..., ::-1]
mean = [103.939, 116.779, 123.68]
image_arr[..., 0] -= mean[0]
image_arr[..., 1] -= mean[1]
image_arr[..., 2] -= mean[2]
image_arr = image_arr.ravel()
print ((image_arr[-10:]))

_test_image = image.load_img(test_image, target_size=(224, 224))
x = image.img_to_array(_test_image)
x = np.expand_dims(x, axis=0)
x = x[..., ::-1]
mean = [103.939, 116.779, 123.68]
x[..., 0] -= mean[0]
x[..., 1] -= mean[1]
x[..., 2] -= mean[2]
x = np.ravel(x)
print(x[-10:])
