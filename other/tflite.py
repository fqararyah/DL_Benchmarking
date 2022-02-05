import numpy as np
import tensorflow as tf
import time

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/Users/qarayah/WD/python/tf/mobilenet_v1_1.0_224_quant/mobilenet_v1_1.0_224_quant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#for i in range(89, 0, -1):
#    print(interpreter.get_tensor(i).shape)

# Test the model on random input data.
input_shape = input_details[0]['shape']
for i in range(10):
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
    t0 = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    print(time.time() - t0)

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)