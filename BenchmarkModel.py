import os
import datetime
import Settings
from tensorflow.python.keras import utils
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import datasets
import time

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

class BenchmarkModel:
    def __init__(self, model_name = '', batch_sizes = [1], inputs_dims = [[32, 32]]):
        self.model_name = model_name
        self.batch_sizes = batch_sizes
        self.inputs_dims = inputs_dims

    def get_metrics(self):
        (train_images, train_labels), (test_images,
                                    test_labels) = datasets.cifar10.load_data()
        
        test_images_preprocessed = test_images / 255.0
        test_images_preprocessed = test_images_preprocessed[0:max(self.batch_sizes) * 10]
        for input_dim in self.inputs_dims:
            if self.model_name == '':
                break
            self.pretrained_model = getattr(tf.keras.applications, \
            self.model_name)(input_shape=(input_dim[0], input_dim[1], 3), weights=None, classes=10)
            self.pretrained_model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(
                            from_logits=True),
                        metrics=['accuracy'])
            with open(Settings.Settings().metrics_file + '_' +self.model_name + '_' + \
                str(datetime.datetime.now()).split('.')[0], 'w') as f:
                for batch_size in self.batch_sizes:
                    f.write('\n----------------\nbatch size: ' + str(batch_size) + '\n----------------\n')
                    #Throughput
                    t0 = time.time()
                    #test_loss, test_acc = pretrained_model.evaluate(test_images,  test_labels)#, verbose=2)
                    tmp = np.argmax(self.pretrained_model.predict(x = test_images_preprocessed, batch_size = batch_size, verbose = 0))
                    f.write("Throughput is: " + str(len(test_images_preprocessed) / (time.time() - t0)) + " images per second.\n")
                    #end throughput

                    #latency
                    avg_time = 0.0
                    avg_time_with_preprocessing = 0.0
                    counter = 0
                    while counter * batch_size < max(self.batch_sizes) * 10:
                        image_batch = test_images[counter * batch_size: (counter + 1) * batch_size + 1]
                        t0_with_preprocessing = time.time()
                        image_batch = image_batch / 255.0
                        t0 = time.time()
                        tmp = np.argmax(self.pretrained_model.predict(x = image_batch, batch_size = batch_size, verbose = 0))
                        avg_time += time.time() - t0
                        avg_time_with_preprocessing += time.time() - t0_with_preprocessing
                        counter += 1

                    avg_time /= counter
                    avg_time_with_preprocessing /= counter
                    f.write("Latency (without processing time) is: " + str(avg_time) + " seconds.\n")
                    f.write("Latency (with processing time) is: " + str(avg_time_with_preprocessing) + " seconds.\n")
                    #end latency
