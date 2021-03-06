import os
import datetime
from tensorflow.python.keras.engine import training

from tensorflow.python.ops.gen_math_ops import arg_max
import Settings
from tensorflow.python.keras import utils
import pathlib

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import datasets
from time import sleep
import time
import statistics

import ssl

VEDLIOT = 1

ssl._create_default_https_context = ssl._create_unverified_context

test_images = []

def load_images():
    global test_images
    if len(test_images) == 0:
        (train_images, train_labels), (test_images,
                                        test_labels) = datasets.cifar10.load_data()
    return test_images

class BenchmarkModel:
    def __init__(self, model_name = '', batch_sizes = [1], inputs_dims = [[32, 32]], bit_widths = [], num_classes = 10):
        self.model_name = model_name
        self.batch_sizes = batch_sizes
        self.inputs_dims = inputs_dims
        self.bit_widths = bit_widths
        self.num_classes = num_classes

    def get_metrics(self):
        if Settings.Settings().power_profile:
            physical_devices = tf.config.list_physical_devices('GPU')
            try:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
            except:
            # Invalid device or cannot modify virtual devices once initialized.
                pass
        #load_images()
        for input_dim in self.inputs_dims:
            test_images = np.random.randint(low =0, high= 256, size = [320, input_dim[0], input_dim[1],\
                 3], dtype=np.uint8)
            test_images_preprocessed = test_images / 255.0
            test_images_preprocessed = test_images_preprocessed[0:max(self.batch_sizes) * 10]
            if self.model_name == '' or self.model_name == Settings.Settings().end_of_file:
                break
            
            self.pretrained_model = getattr(tf.keras.applications, \
            self.model_name)(input_shape=(input_dim[0], input_dim[1], 3), weights=None, classes=self.num_classes)

            self.pretrained_model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(
                            from_logits=True),
                        metrics=['accuracy'])
            for bit_width in self.bit_widths:
                """ #currently only 16 bit float is supported
                tflite_models_dir = pathlib.Path(Settings.Settings().tflite_folder)
                tflite_models_dir.mkdir(exist_ok=True, parents=True)
                if bit_width == 32:
                    tflite_model_file = tflite_models_dir/(self.model_name+"model_quant_32.tflite")
                elif bit_width == 16:
                    tflite_model_file = tflite_models_dir/(self.model_name+"model_quant_16.tflite")
                
                if not tflite_model_file.exists():
                    converter = tf.lite.TFLiteConverter.from_keras_model(self.pretrained_model)
                    tflite_model = converter.convert()
                    
                    if bit_width == 32:
                        converter.target_spec.supported_types = [tf.float32]
                        print(32)
                    elif bit_width == 16:
                        converter.optimizations = [tf.lite.Optimize.DEFAULT]
                        converter.target_spec.supported_types = [tf.float16]
                        print(16)
                    else:
                        break
                    tflite_quantized_model = converter.convert()
                    tflite_model_file.write_bytes(tflite_quantized_model)

                interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))

                self.get_metrics_quantized(input_dim, test_images_preprocessed, test_images, bit_width, interpreter)
 """            
                print(bit_width)
                if bit_width == 32:
                    if VEDLIOT:
                        self.get_metrics_32_ved(input_dim, test_images_preprocessed, test_images)
                    else:
                        self.get_metrics_32(input_dim, test_images_preprocessed, test_images)

    def get_metrics_32(self, input_dim, test_images_preprocessed, test_images):
        settings_obj = Settings.Settings()
        with open(settings_obj.metrics_file + '_' +self.model_name + '_32_' + str(input_dim[0]) + 'x' + \
            str(input_dim[1]) + '_' + str(datetime.datetime.now()).split('.')[0], 'w') as f:
            for batch_size in self.batch_sizes:
                
                with open (settings_obj.status_file_name, 'w') as inner_f:
                    inner_f.write(self.model_name + '_32_' +  str(input_dim[0]) + 'x' + str(input_dim[1]) + '_' + \
                        str(batch_size))

                f.write('\n----------------\nbatch size: ' + str(batch_size) + '\n----------------\n')
                #this is to load the model
                """ tmp = np.argmax(self.pretrained_model.predict(x = test_images[max(self.batch_sizes)*10:max(self.batch_sizes)*20]/255.0, batch_size = \
                    batch_size, verbose = 0))
                #Throughput
                t0 = time.time()
                #test_loss, test_acc = pretrained_model.evaluate(test_images,  test_labels)#, verbose=2)
                tmp = np.argmax(self.pretrained_model.predict(x = test_images_preprocessed, batch_size = batch_size, verbose = 0), 1)
                f.write("Execution time is: " + str((time.time() - t0) / len(test_images_preprocessed)) + "seconds.\n") """
                #end throughput

                #latency
                avg_time = 0.0
                #avg_time_with_preprocessing = 0.0
                counter = 0
                #tf.profiler.experimental.start('./out')
                
                while counter * batch_size < max(self.batch_sizes) * 10:
                    image_batch = test_images[counter * batch_size: (counter + 1) * batch_size]
                    #t0_with_preprocessing = time.time()
                    tmp = -1
                    t0 = time.time()
                    image_batch = image_batch / 255.0
                    tmp = np.argmax(self.pretrained_model(image_batch, training = False))
                    if tmp != -1 and counter > 0:
                        avg_time += time.time() - t0
                    #avg_time_with_preprocessing += time.time() - t0_with_preprocessing
                    counter += 1
                
                #tf.profiler.experimental.stop()

    def get_metrics_32_ved(self, input_dim, test_images_preprocessed, test_images):
        settings_obj = Settings.Settings()
        if settings_obj.power_profile:
            physical_devices = tf.config.list_physical_devices('GPU')
            try:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
            except:
            # Invalid device or cannot modify virtual devices once initialized.
                pass
        with open(settings_obj.metrics_file + '_' +self.model_name + '_32_' + str(input_dim[0]) + 'x' + \
            str(input_dim[1]) + '_ved_' + str(datetime.datetime.now()).split('.')[0], 'w') as f:
                                
            avg_lats = []
            avg_execs = []
            batch_size = 1
            for i in range(0, 1005):
                tmp = -1
                image_batch = np.random.randint(low =0, high= 256, size = [batch_size, 224, 224, 3], dtype=np.uint8)
                t0 = time.time()
                image_batch = image_batch / 255.0
                t1 = time.time()
                tmp = np.argmax(self.pretrained_model(image_batch, training = False))
                if settings_obj.power_profile:
                    with open (settings_obj.status_file_name, 'w') as inner_f:
                        inner_f.write('invalid')
                        sleep(0.1)
                if tmp != -1 and i >= 5:
                    avg_execs.append(1000 * (time.time() - t1) / batch_size)
                    avg_lats.append(1000 * (time.time() - t0) / batch_size)
            
            avg_lats.sort()
            avg_execs.sort()
            f.write("Mean latency is:\t\t" + str(sum(avg_lats) / 1000) + " ms.\n")
            f.write("Median latency is:\t\t" + str(avg_lats[500]) + " ms.\n")
            f.write("STD latency is:\t\t" + str(statistics.stdev(avg_lats)) + " ms.\n")

            f.write("Mean exec-time is:\t\t" + str(sum(avg_execs) / 1000) + " ms.\n")
            f.write("Median exec-time is:\t\t" + str(avg_execs[500]) + " ms.\n")
            f.write("STD exec-time is:\t\t" + str(statistics.stdev(avg_execs)) + " ms.\n")
            

