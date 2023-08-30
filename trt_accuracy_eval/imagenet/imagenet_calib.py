import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import os

from PIL import Image

import common

MODEL_NAME = common.MODEL_PATH.split('/')[-1].split('.')[0].lower()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
class ModelData(object):
    INPUT_SHAPE = (3, 224, 224)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32

class ResNet50EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file, batch_size, total_images):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_index = 0
        self.test_images = common.locate_images(common.CALIB_DATASET_PATH)
        self.limit = len(self.test_images)
        self.limit = total_images
        if total_images == -1:
            self.limit = len(self.test_images)

        # Allocate enough memory for a whole batch.
        nbytes = np.zeros(shape=(2244,224,3)).nbytes
        self.device_input = cuda.mem_alloc(nbytes)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.limit:
            return None

        current_batch = int(self.current_index / self.batch_size)
        print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        ###
        test_image = self.test_images[self.current_index]
        image = Image.open(test_image)
        image = image.convert('RGB')
        c, h, w = ModelData.INPUT_SHAPE
        resized_image = image.resize((w, h), Image.ANTIALIAS)
        np_array = np.asarray(resized_image)
        #print(np_array.shape)
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
            
        #print(image_arr.shape)
        batch = np.ravel(image_arr)
        
        ###
        # loaded as RGB, convert to BGR
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def get_engine(onnx_file_path, engine_file_path, batch_size, calibrator=None):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine(calibrator=None):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            # Parse model file
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            print('Completed parsing of ONNX file')
            
            print('Network inputs:')
            for i in range(network.num_inputs):
                tensor = network.get_input(i)
                print(tensor.name, trt.nptype(tensor.dtype), tensor.shape)

            network.get_input(0).shape = [batch_size, 224, 224, 3]

            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 20  # 256MiB
            if calibrator:
                config.set_flag(trt.BuilderFlag.INT8)
                config.int8_calibrator = calibrator

            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_engine(network, config)
            print("Completed creating Engine. Writing file to: {}".format(engine_file_path))

            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(calibrator)