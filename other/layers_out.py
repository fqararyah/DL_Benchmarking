import sys
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.keras import datasets
import datetime
import ssl


ssl._create_default_https_context = ssl._create_unverified_context

pretrained_model = tf.keras.applications.MobileNet()
print("** Model architecture **")
pretrained_model.summary()

image_batch = np.random.randint(low =0, high= 256, size = [1, 224, 224, 3], dtype=np.uint8)


counter = 0
#while counter < 100:
image = image_batch[counter]
image = np.expand_dims(image, axis=0).astype(np.float32)
#predictions = pretrained_model.predict(x = image, batch_size = 1, verbose = 0)

intermediate_outputs = []
sparcity_ratio = []
for i in range(0, len(pretrained_model.layers)):
    layer_name = pretrained_model.layers[i].name
    intermediate_layer_model = tf.keras.Model(inputs=pretrained_model.input,
                                       outputs=pretrained_model.get_layer(layer_name).output)
    intermediate_outputs.append(intermediate_layer_model(image).numpy())
    #print(i, "---------------------------")
    sparcity_ratio.append(1 - np.count_nonzero(intermediate_outputs[-1]) / np.prod(np.shape(intermediate_outputs[-1])))

for i in range(0, len(pretrained_model.layers)):
    print(pretrained_model.layers[i].name ,sparcity_ratio[i])

""" if np.sum(predictions - intermediate_outputs[-1]) != 0:
    print("Error")
else:
    np.set_printoptions(threshold=sys.maxsize)
    with open('/Users/qarayah/WD/python/benchmarking/other/out/' + pretrained_model.name + '_' +str(datetime.datetime.now()).split('.')[0], 'w') as f:
        for layer_output in intermediate_outputs:
            f.write(str(layer_output) + '\n') """