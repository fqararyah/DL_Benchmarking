import onnx
import warnings
from onnx_tf.backend import prepare
import numpy as np
import tensorflow as tf
import time
import statistics

warnings.filterwarnings('ignore') # Ignore all the warning messages in this tutorial
model = onnx.load('/home/fareed/Downloads/yolov4.onnx') # Load the ONNX file
out_file = './onnx_out/yolov4.txt'
tf_rep = prepare(model)

print(tf_rep.inputs) # Input nodes to the model
print('-----')
print(tf_rep.outputs) # Output nodes from the model
print('-----')
print(tf_rep.tensor_dict) # All nodes in the model

test_images = np.random.randint(low =0, high= 256, size = [16, 416, 416, 3], dtype=np.uint8)
test_images = test_images / 255.0
test_images = tf.cast(test_images, dtype='float32')
avg_execs = []
avg_lats = []
num_iters = 1000
for i in range(0, num_iters + 5):
    test_images = np.random.randint(low =0, high= 256, size = [16, 416, 416, 3], dtype=np.uint8)
    t0 = time.time()
    test_images = tf.cast(test_images, dtype='float32')
    test_images = test_images / 255.0
    t1 = time.time()
    ret = tf_rep.run(test_images)
    if i >= 5:
        avg_execs.append((time.time() - t1) / 16)
        avg_lats.append((time.time() - t0) /16)

avg_lats.sort()
avg_execs.sort()

with open(out_file, 'w') as f:
    f.write("Mean latency is: " + str( 1000 * sum(avg_lats) / num_iters) + " ms.\n")
    f.write("Median latency is: " + str(1000 * avg_lats[int(num_iters / 2)]) + " ms.\n")
    f.write("STD latency is: " + str(1000 * statistics.stdev(avg_lats)) + " ms.\n")

    f.write("Mean exec-time is: " + str(1000 * sum(avg_execs) / num_iters) + " ms.\n")
    f.write("Median exec-time is: " + str(1000 * avg_execs[int(num_iters / 2)]) + " ms.\n")
    f.write("STD exec-time is: " + str(1000 * statistics.stdev(avg_execs)) + " ms.\n")
