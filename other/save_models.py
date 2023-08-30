import os
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import datasets
import ssl
import time


ssl._create_default_https_context = ssl._create_unverified_context

pretrained_model = tf.keras.applications.VGG16()

model_dir = "./models/VGG16"
pretrained_model.save(model_dir)