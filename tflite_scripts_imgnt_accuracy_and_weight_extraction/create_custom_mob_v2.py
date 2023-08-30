import tensorflow as tf
from keras.models import Model
import tensorflow.keras.applications as applications

model = applications.MobileNetV2()
inp = model.input
new_layer = tf.keras.layers.Dense(2, activation='softmax')
out = new_layer(model.layers[-2].output)

my_model = Model(inp, out)

my_model.save('mobilenet_v2.h5')