#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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

import common

import json

import imagenet_calib

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess_input

class ModelData(object):
    MODEL_PATH = "ResNet50.onnx"
    INPUT_SHAPE = (3, 224, 224)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32


# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
MODEL_NAME = common.MODEL_PATH.split('/')[-1].split('.')[0].lower()

# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(common.EXPLICIT_BATCH)
    config = builder.create_builder_config()

    #new
    if common.PRECISION == '8':
        print('***************************')
        print('**********8***************')
        print('***************************')
        calibration_cache = "mnist_calibration.cache"
        calibrator = imagenet_calib.ResNet50EntropyCalibrator(cache_file=calibration_cache, batch_size=1, total_images=5000)
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
    elif common.PRECISION == '16':
        print('***************************')
        print('**********16***************')
        print('***************************')
        config.set_flag(trt.BuilderFlag.FP16)
    #end_new

    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.max_workspace_size = common.GiB(1)
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(model_file, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    engine_file_path = engine_file_path =  common.MODEL_PATH.split('/')[-1].split('.')[0].lower() + str(common.PRECISION) + ".trt"

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
    else:
        plan = builder.build_serialized_network(network, config)
        engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(plan)
        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(plan)
    
    return engine


def load_normalized_test_case(test_image, pagelocked_buffer):
    # Converts the input image to a CHW Numpy array
    def normalize_image(test_image):
        # Resize, antialias and transpose the image to CHW.
        source = 'EMBDL'
        if source != 'EMBDL':
            _test_image = image.load_img(test_image, target_size=(224, 224))
            c, h, w = ModelData.INPUT_SHAPE
            resized_image = _test_image.resize((w, h), Image.ANTIALIAS)
            np_array = np.asarray(resized_image)
            if(np_array.ndim == 2):
                np_array = np.repeat(np_array[:, :, np.newaxis], 3, axis=2)
            image_arr = (
                np_array
                .transpose([2, 0, 1])
                .astype(trt.nptype(ModelData.DTYPE))
                .ravel()
            )
            # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
            return (image_arr / 255.0 - 0.45) / 0.225
        else:
            _test_image = image.load_img(test_image, target_size=(224, 224))
            x = image.img_to_array(_test_image)
            x = np.expand_dims(x, axis=0)
            if 'resnet' in MODEL_NAME:
                x = resnet_preprocess_input(x)
            elif 'mobilenet' in MODEL_NAME:
                x = mobilenet_preprocess_input(x)
            x = np.ravel(x)
            return x

    # Normalize the image and copy to pagelocked memory.
    #np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
    np.copyto(pagelocked_buffer, normalize_image(test_image))
    return test_image


def main():
    # Set the data path to the directory that contains the trained models and test images for inference.
    # _, data_files = common.find_sample_data(
    #     description="Runs a ResNet50 network with a TensorRT inference engine.",
    #     subfolder="resnet50",
    #     find_files=[
    #         "binoculars.jpeg",
    #         "reflex_camera.jpeg",
    #         "tabby_tiger_cat.jpg",
    #         ModelData.MODEL_PATH,
    #         "class_labels.txt",
    #     ],
    # )
    # Get test images, models and labels.
    test_images = common.locate_images(common.DATASET_PATH)
    #data_files[0:3]
    onnx_model_file = common.MODEL_PATH
    #, labels_file = data_files[3:]
    #labels = open(common.LABELS_PATH, "r").read().split("\n")

    # Build a TensorRT engine.
    engine = build_engine_onnx(onnx_model_file)
    # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
    # Allocate buffers and create a CUDA stream.
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    # Contexts are used to perform inference.
    context = engine.create_execution_context()
  
    # Opening JSON file
    #f = open(common.GROUND_TRUTH_PATH)
    #ground_truth = json.load(f)

    #for entry in ground_truth:
    #    print(entry[0])
    
    print(len(test_images))
    bad_count = 0
    prediction_dict_list = []
    for i in range(len(test_images)):
        # Load a normalized test case into the host input page-locked buffer.
        test_image = test_images[i]
        #print(test_image)
        #try:
        test_case = load_normalized_test_case(test_image, inputs[0].host)
        # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
        # probability that the image corresponds to that label
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        # We use the highest probability as our prediction. Its index corresponds to the predicted label.
        #pred = labels[np.argmax(trt_outputs[0])]
        indices = np.argsort(trt_outputs[0])[-5:]
        indices = np.flip(indices)
        prediction_dict = {"dets":indices.tolist(), "image": test_image.split('/')[-1]}
        prediction_dict_list.append(prediction_dict)
        if indices[0] - np.argmax(trt_outputs[0]) != 0 and trt_outputs[0][np.argmax(trt_outputs[0])] != trt_outputs[0][indices[0]]:
            print('something is wrong:', i)
        #except:
        #    bad_count += 1

    json_object = json.dumps(prediction_dict_list)
    print('bad images:',bad_count)

    with open(MODEL_NAME + '_' + common.PRECISION + "_predictions.json", "w") as outfile:
        outfile.write(json_object)



if __name__ == "__main__":
    main()

#python3 ./onnx_resnet50.py -d/home/fareed/Downloads/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1/data/resnet50