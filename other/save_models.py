import os
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import datasets
import time

pretrained_model = tf.keras.applications.ResNet50()

model_dir = "./models/ResNet50"
pretrained_model.save(model_dir)