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

import time
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


class ModelData(object):
    INPUT_SHAPE = (3, 224, 224)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32


# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
MODEL_NAME = common.MODEL_PATH.split('/')[-1].split('.')[0].lower()


class MyProfiler(trt.IProfiler):
    profiling_dict = {}

    def __init__(self):
        trt.IProfiler.__init__(self)

    def report_layer_time(self, layer_name, ms):
        self.profiling_dict[layer_name] = ms

# The Onnx path is used for Onnx models.


def build_engine_onnx(model_file):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(common.EXPLICIT_BATCH)
    config = builder.create_builder_config()

    # new
    if common.PRECISION == '8':
        print('***************************')
        print('**********8***************')
        print('***************************')
        calibration_cache = "mnist_calibration.cache"
        calibrator = imagenet_calib.ResNet50EntropyCalibrator(
            cache_file=calibration_cache, batch_size=32, total_images=-1)
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
    elif common.PRECISION == '16':
        print('***************************')
        print('**********16***************')
        print('***************************')
        config.set_flag(trt.BuilderFlag.FP16)
    # end_new

    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.max_workspace_size = common.GiB(1)
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(model_file, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    engine_file_path = common.MODEL_PATH.split(
        '/')[-1].split('.')[0].lower() + '_' + str(common.PRECISION) + ".trt"

    print('***********', engine_file_path)
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
    def normalize_image(image):
        # Resize, antialias and transpose the image to CHW.
        image = image.convert('RGB')
        c, h, w = ModelData.INPUT_SHAPE
        resized_image = image.resize((w, h), Image.ANTIALIAS)
        np_array = np.asarray(resized_image)
        # print(np_array.shape)
        image_arr = (
            np_array
            .astype(trt.nptype(ModelData.DTYPE))
        )
        # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
        if 'resnet' in MODEL_NAME.lower():
            image_arr = image_arr[..., ::-1]
            mean = [103.939, 116.779, 123.68]
            image_arr[..., 0] -= mean[0]
            image_arr[..., 1] -= mean[1]
            image_arr[..., 2] -= mean[2]

        image_arr = image_arr.ravel()
        # print(image_arr.shape)
        return image_arr

    # Normalize the image and copy to pagelocked memory.
    t1 = time.time()
    np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
    return test_image, t1

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
    # data_files[0:3]
    onnx_model_file = common.MODEL_PATH
    # , labels_file = data_files[3:]
    #labels = open(common.LABELS_PATH, "r").read().split("\n")

    # Build a TensorRT engine.
    engine = build_engine_onnx(onnx_model_file)
    # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
    # Allocate buffers and create a CUDA stream.
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    # Contexts are used to perform inference.
    context = engine.create_execution_context()
    context.profiler = MyProfiler()

    # Opening JSON file
    #f = open(common.GROUND_TRUTH_PATH)
    #ground_truth = json.load(f)

    # for entry in ground_truth:
    #    print(entry[0])

    print(len(test_images))
    images_to_test = 10000
    power_measurement = True
    bad_count = 0
    prediction_dict_list = []
    avg_time = 0
    for i in range(len(test_images)):
        # Load a normalized test case into the host input page-locked buffer.
        if i == images_to_test:
            break
        test_image = test_images[i]
        # print(test_image)
        # try:
        if i == 0 or not power_measurement:
            test_case, t1 = load_normalized_test_case(
                test_image, inputs[0].host)
        t1 = time.time()
        # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
        # probability that the image corresponds to that label
        trt_outputs = common.do_inference_v2(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        # We use the highest probability as our prediction. Its index corresponds to the predicted label.
        #pred = labels[np.argmax(trt_outputs[0])]
        t2 = time.time()
        indices = np.argsort(trt_outputs[0])[-5:]
        if i >= 5:
            avg_time += t2 - t1
        indices = np.flip(indices)
        prediction_dict = {"dets": indices.tolist(
        ), "image": test_image.split('/')[-1]}
        prediction_dict_list.append(prediction_dict)
        if indices[0] - np.argmax(trt_outputs[0]) != 0 and trt_outputs[0][np.argmax(trt_outputs[0])] != trt_outputs[0][indices[0]]:
            print('something is wrong:', i)
        # except:
        #    bad_count += 1
    print('###############################')
    sum = 0
    print_layer_name = False
    print_layer_index = False
    i = 0
    for layer_name, layer_time in MyProfiler.profiling_dict.items():
        if print_layer_name and print_layer_index:
            print(i, layer_name, layer_time)
            i += 1
        elif print_layer_name:
            print(layer_name, layer_time)
            i += 1
        elif ('conv' in layer_name.lower() or 'depthwise' in layer_name.lower()) and 'reformatting' not in layer_name.lower() \
                and 'pwn()' not in layer_name.lower():
            if print_layer_index:
                print(i, layer_time)
            else:
                print(layer_time)

            i += 1

        sum += layer_time

    i = 0

    print('sum', sum)
    print('###############################')

    json_object = json.dumps(prediction_dict_list)
    print('bad images:', bad_count)

    avg_time /= (images_to_test - 5)
    print('average inference time:', avg_time * 1000, ' ms')

    with open(MODEL_NAME + '_' + common.PRECISION + "_predictions.json", "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    main()

# python3 ./onnx_resnet50.py -d/home/fareed/Downloads/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1/data/resnet50
