#!/usr/bin/env python3
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

from __future__ import print_function
from pickle import NONE

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw

from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

import os
import json

import common

import time

TRT_LOGGER = trt.Logger()


def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color="blue"):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    draw = ImageDraw.Draw(image_raw)
    print(bboxes, confidences, categories)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12), "{0} {1:.2f}".format(all_categories[category], score), fill=bbox_color)

    return image_raw


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.max_workspace_size = 1 << 28  # 256MiB
            if PRECISION == '16':
                print('***************************')
                print('**********16***************')
                print('***************************')
                config.set_flag(trt.BuilderFlag.FP16)
            if PRECISION == '8':
                print('***************************')
                print('**********8***************')
                print('***************************')
                profile = builder.create_optimization_profile()
                from coco_calib import YOLOEntropyCalibrator
                config.set_flag(trt.BuilderFlag.INT8)
                config.int8_calibrator = YOLOEntropyCalibrator(
                    CALIB_DATASET_PATH, (608, 608),
                    'calib_%s.bin' % engine_file_path)
                #config.set_calibration_profile(profile)
            #builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov4_to_onnx.py first to generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov4.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, 608, 608]
            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def locate_images(path):
    image_list = []
    # dirs=directories
    for (root, dirs, file) in os.walk(path):
        for f in file:
            if '.jpg' in f:
                image_list.append(os.path.abspath(os.path.join(path, f)))
                #print(image_list[-1])
    return image_list

DATASET_PATH = ''
MODEL_PATH = ''
LABELS_PATH = ''
GROUND_TRUTH_PATH = ''
PRECISION = 32
CALIB_DATASET_PATH = ''

def read_settings():
    global DATASET_PATH, MODEL_PATH, LABELS_PATH, GROUND_TRUTH_PATH, PRECISION, CALIB_DATASET_PATH
    with open('./settings.txt', 'r') as f:
        for line in f:
            line = line.replace('\n', '').replace(' ', '')
            if 'calib_dataset_path' in line.lower():
                CALIB_DATASET_PATH = line.split('::')[1]
            elif 'dataset_path' in line.lower():
                DATASET_PATH = line.split('::')[1]
            elif 'model_path' in line.lower():
                MODEL_PATH = line.split('::')[1]
            elif 'labels_path' in line.lower():
                LABELS_PATH = line.split('::')[1]
            elif 'ground_truth_path' in line.lower():
                GROUND_TRUTH_PATH = line.split('::')[1]
            elif 'precision':
                PRECISION = line.split('::')[1]

def main():
    """Create a TensorRT engine for ONNX-based YOLOv4-608 and run inference."""
    read_settings()
    MODEL_NAME = MODEL_PATH.split('/')[-1].split('.')[0].lower()
    # Try to load a previously generated YOLOv4-608 network graph in ONNX format:
    onnx_file_path = MODEL_PATH
    engine_file_path = "yolov4_" + PRECISION + ".trt"
    # Download a dog image and save it to the following file path:
    image_paths = locate_images(DATASET_PATH)
    prediction_dict_list = []
    step = 0
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        for input_image_path in image_paths: 
            #print(input_image_path)
            # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
            input_resolution_yolov4_HW = (608, 608)
            # Create a pre-processor object by specifying the required input resolution for YOLOv4
            preprocessor = PreprocessYOLO(input_resolution_yolov4_HW)
            # Load an image from the specified input path, and return it together with  a pre-processed version
            image_raw, image = preprocessor.process(input_image_path)
            # Store the shape of the original input image in WH format, we will need it for later
            shape_orig_WH = image_raw.size

            # Output shapes expected by the post-processor
            output_shapes = [(1, 255, 76, 76), (1, 255, 38, 38),(1, 255, 19, 19)]
            # Do inference with TensorRT
            trt_outputs = []
            
            t0 = time.time()
            inputs, outputs, bindings, stream = common.allocate_buffers(engine)
            # Do inference
            #print("Running inference on image {}...".format(input_image_path))
            # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
            inputs[0].host = image
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

            t1 = time.time()
            print(t1 - t0)
            # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
            trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

            postprocessor_args = {
                "yolo_masks": [(0, 1, 2), (3, 4, 5), (6, 7, 8)],  # A list of 3 three-dimensional tuples for the YOLO masks
                "yolo_anchors": [
                    (12, 16), (19, 36), (40, 28), (36, 75), (76, 55), (72, 146), (142, 110), (192, 243), (459, 401)
                    # (10, 13),
                    # (16, 30),
                    # (33, 23),
                    # (30, 61),
                    # (62, 45),  # A list of 9 two-dimensional tuples for the YOLO anchors
                    # (59, 119),
                    # (116, 90),
                    # (156, 198),
                    # (373, 326),
                ],
                "obj_threshold": 0.00001,  # Threshold for object coverage, float value between 0 and 1 , fareed: it was 0.6
                "nms_threshold": 0.5,  # Threshold for non-max suppression algorithm, float value between 0 and 1, , fareed: it was 0.5
                "yolo_input_resolution": input_resolution_yolov4_HW,
            }
            
            postprocessor = PostprocessYOLO(**postprocessor_args)

            # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
            boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH))

            if boxes is not None:
                for (box, _class, score) in zip(boxes, classes, scores): 
                    prediction_dict = {"image_id":int(input_image_path.split('/')[-1].split('.')[0]), "category_id": int(_class)\
                        , "bbox": box.tolist(), "score": float(score)}
                    prediction_dict_list.append(prediction_dict)
                    #print(prediction_dict)
            print(step)
            step += 1

            #if step == 10:
            #    break

    json_object = json.dumps(prediction_dict_list)

    with open(MODEL_NAME + '_' + str(PRECISION) + "_predictions.json", "w") as outfile:
        outfile.write(json_object)
        # Draw the bounding boxes onto the original input image and save it as a PNG file
        #obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, ALL_CATEGORIES)
        #output_image_path = "dog_bboxes.png"
        #obj_detected_img.save(output_image_path, "PNG")
        #print("Saved image with bounding boxes of detected objects to {}.".format(output_image_path))


if __name__ == "__main__":
    main()