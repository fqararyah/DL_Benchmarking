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
    def __init__(self, model_name = '', batch_sizes = [1]):
        self.model_name = model_name
        self.batch_sizes = batch_sizes
        self.set_model()

    def set_model(self):
        if self.model_name != '':
            self.pretrained_model = getattr(tf.keras.applications, \
            self.model_name)(input_shape=(32, 32, 3), weights=None, classes=10)

    def get_metrics(self):
        (train_images, train_labels), (test_images,
                                    test_labels) = datasets.cifar10.load_data()

        self.pretrained_model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(
                            from_logits=True),
                        metrics=['accuracy'])
        
        test_images = test_images / 255.0
        test_images = test_images[0:max(self.batch_sizes * 10)]
        with open(Settings.Settings().metrics_file + self.model_name + str(datetime.datetime.now()).split('.')[0], 'w') as f:
            for batch_size in self.batch_sizes:
                f.write('\n----------------\nbatch size: ' + str(batch_size) + '\n----------------\n')
                #Throughput
                t0 = time.time()
                #test_loss, test_acc = pretrained_model.evaluate(test_images,  test_labels)#, verbose=2)
                tmp = np.argmax(self.pretrained_model.predict(x = test_images, batch_size = batch_size, verbose = 0))
                f.write("Throughput is: " + str(len(test_images) / (time.time() - t0)) + " images per second.\n")
                #end throughput

                #latency
                avg_time = 0.0
                counter = 0
                while counter * batch_size < max(self.batch_sizes) * 10:
                    image = test_images[counter * batch_size: counter * (batch_size + 1)]
                    t0 = time.time()
                    image = np.expand_dims(image, axis=0).astype(np.float32)
                    tmp = np.argmax(self.pretrained_model.predict(x = image, batch_size = batch_size, verbose = 0))
                    avg_time += time.time() - t0
                    counter += 1

                avg_time /= counter
                f.write("Latency is: " + str(avg_time) + " seconds.\n")
                #end latency
