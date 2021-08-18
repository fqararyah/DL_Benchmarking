import os
import tempfile

from matplotlib import pyplot as plt
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

pretrained_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

tf.profiler.experimental.start('./logs')
test_loss, test_acc = pretrained_model.fit(train_images, train_labels, epochs=1,
                        validation_data=(test_images, test_labels))
tf.profiler.experimental.stop()

t0 = time.time()
test_loss, test_acc = pretrained_model.evaluate(
    test_images,  test_labels, verbose=2)
print("The time taken is", time.time() - t0)