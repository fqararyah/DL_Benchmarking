import os
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import datasets
import time

pretrained_model = tf.keras.applications.ResNet50(input_shape=(32, 32, 3), weights=None, classes=10)

(train_images, train_labels), (test_images,
                               test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = train_images[0:1000]
train_labels = train_labels[0:1000]
test_images = test_images[0:100]
test_labels = test_labels[0:100]
pretrained_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

log_dir = "./logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')
pretrained_model.fit(x=train_images, 
          y=train_labels, 
          epochs=1, 
          validation_data=(test_images, test_labels), callbacks=[tensorboard_callback])

""" t0 = time.time()
test_loss, test_acc = pretrained_model.evaluate(
    test_images,  test_labels, verbose=2)
print("The time taken is", time.time() - t0) """