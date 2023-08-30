import os
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import datasets
import ssl
import time


ssl._create_default_https_context = ssl._create_unverified_context

pretrained_model = tf.keras.applications.EfficientNetB0()

# Normalize pixel values to be between 0 and 1
test_images = np.random.randint(low =0, high= 256, size = [1, 224, 224,\
                 3], dtype=np.uint8)


tmp = np.argmax(pretrained_model(test_images, training = False))

print(tmp)
