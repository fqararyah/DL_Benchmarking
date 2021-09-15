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

model_dir = "./models/EfficientNetB0"
pretrained_model.save(model_dir)