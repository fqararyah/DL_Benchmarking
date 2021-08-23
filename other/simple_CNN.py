import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import pathlib
import numpy as np
import time

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

(train_images, train_labels), (test_images,
                               test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


def train():
    model = models.Sequential()
    model.add(layers.Conv2D(
        32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

    return model


model_dir = "./models/simple_CNN"

try:
    model = models.load_model(model_dir)
except OSError:
    model = train()
    model.save(model_dir)

print("** Model architecture **")
model.summary()

#t0 = time.time()
test_loss, test_acc = model.evaluate(
    test_images,  test_labels, verbose=2)

#print("*****************************")
#print(time.time() - t0)
#print("*****************************")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_types = [tf.float16]

tflite_fp16_model = converter.convert()

tflite_models_dir = pathlib.Path("./quant_models/simple_CNN/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

#tflite_model_fp16_file = tflite_models_dir/"mnist_model_quant_f16.tflite"

tflite_model_fp16_file = tflite_models_dir/"mnist_model_quant.tflite"
tflite_model_fp16_file.write_bytes(tflite_fp16_model)

interpreter_fp16 = tf.lite.Interpreter(model_path=str(tflite_model_fp16_file))
interpreter_fp16.allocate_tensors()


# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_model(interpreter):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    index = 0
    accomulate_accuraccy = 0
    accomulate_t1 = 0
    accomulate_t2 = 0
    for test_image in test_images:
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_image)

        t0 = time.time()
        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)
        accomulate_t1 += time.time() - t0
        #print("a", time.time() - t0)

        """ t0 = time.time()
        _, one_acc = model.evaluate(test_image,  test_labels[index], verbose=0)
        accomulate_accuraccy += one_acc
        accomulate_t2 += time.time() - t0 """
        #print("b", time.time() - t0)
        index += 1

    # Compare prediction results with ground truth labels to calculate accuracy.
    accurate_count = 0
    for index in range(len(prediction_digits)):
        if prediction_digits[index] == test_labels[index]:
            accurate_count += 1
    quant_accuracy = accurate_count * 1.0 / len(prediction_digits)
    accuracy = accomulate_accuraccy / len(prediction_digits)

    return quant_accuracy, accuracy, accomulate_t1, accomulate_t2

quant_test_acc, _, accomulate_t1, accomulate_t2 = evaluate_model(interpreter_fp16)

print("model accuracy is:", test_acc, ", net time is:", accomulate_t2)
print("Quantized model accuracy is:", quant_test_acc, ", net time is:", accomulate_t1)
