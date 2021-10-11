import sys
from operator import mod
import numpy as np
from numpy.lib.type_check import imag
from tensorflow import keras
from tensorflow.keras import layers, models

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", use_bias=False),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", use_bias=False),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

def train():
    batch_size = 128
    epochs = 5

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

model_dir = "/Users/qarayah/WD/python/benchmarking/other/models/simple_CNN_mnist"

try:
    model = models.load_model(model_dir)
except OSError:
    train()
    model.save(model_dir)

#test_loss, test_acc = model.evaluate(x_test, y_test)
#print("test accuracy is:", test_acc)

image = x_test[0]

#from matplotlib import pyplot
#pyplot.imshow(image, cmap=pyplot.get_cmap('gray'))
#pyplot.show()

image = np.expand_dims(image, axis=0).astype(np.float32)

intermediate_outputs = []
for i in range(0, len(model.layers)):
    layer_name = model.layers[i].name
    intermediate_layer_model = keras.Model(inputs=model.input,
                                       outputs=model.get_layer(layer_name).output)
    intermediate_outputs.append(intermediate_layer_model(image).numpy())
    #print(i, "---------------------------")

predictions = model.predict(x = image, batch_size = 1, verbose = 0)
""" for i in range(0, 26):
    strr = ''
    for j in range(0, 26):
        strr += ' ' + str(intermediate_outputs[0][0][i][j][0])
    print(strr) """
if np.sum(predictions - intermediate_outputs[-1]) != 0:
    print("Error")
else:
    np.set_printoptions(threshold=sys.maxsize)
    image = image.squeeze()
    print(image.shape)
    
    f1 = open('/Users/qarayah/WD/python/benchmarking/other/out/simple_mnist_input.txt', 'w')
    np.savetxt(f1, image)

    with open('/Users/qarayah/WD/python/benchmarking/other/out/simple_mnist_layer1.txt', 'w') as f2:
        for layer in range(0, len(model.weights)):
            layer_weights = model.weights[layer].numpy()
            #f2.write(str(layer_weights.shape).replace('(', '').replace(')', '').replace(',', '') + "\n")
            for filter in range(0, layer_weights.shape[3]):
                for depth in range(0, layer_weights.shape[2]):
                    for row in range(0, layer_weights.shape[0]):
                        for col in range(layer_weights.shape[1]):
                            f2.write(str(layer_weights[row][col][depth][filter]) + ' ')
                    f2.write("\n")
            break
    
    with open('/Users/qarayah/WD/python/benchmarking/other/out/simple_mnist_out1.txt', 'w') as f2:
        for layer_output in intermediate_outputs:
            #f2.write(str(layer_output.shape).replace('(', '').replace(')', '').replace(',', '') + '\n')
            for depth in range(0, layer_output.shape[3]):
                    for row in range(0, layer_output.shape[1]):
                        for col in range(layer_output.shape[2]):
                            f2.write(str(layer_output[0][row][col][depth]) + ' ')
                        f2.write("\n")
                    f2.write("\n")
            break

