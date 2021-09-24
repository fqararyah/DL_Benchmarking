import time
from numpy.lib.type_check import imag
import tensorflow as tf
import numpy as np
import statistics

def load_pb(path_to_pb):
    with tf.io.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

#test_images = tf.cast(test_images, dtype='float32')

graph = load_pb('/Users/qarayah/Downloads/MobileNetV3Small.pb')
out_file = './pb_out/yolov4.txt'

#print(graph.get_operations())
input = graph.get_tensor_by_name('input_1:0')
output = graph.get_tensor_by_name('Predictions/Softmax:0')

sess = tf.compat.v1.Session(graph=graph)


avg_execs = []
avg_lats = []
num_iters = 1000
batch_size = 1
for i in range(0, num_iters + 5):
    test_images = np.random.randint(low =0, high= 256, size = [16, 416, 416, 3], dtype=np.uint8)
    t0 = time.time()
    test_images = test_images / 255.0
    t1 = time.time()
    sess.run(output, feed_dict={input: test_images})
    if i >= 5:
        avg_execs.append((time.time() - t1) /  batch_size)
        avg_lats.append((time.time() - t0) /batch_size)

avg_lats.sort()
avg_execs.sort()

with open(out_file, 'w') as f:
    f.write("Mean latency is: " + str( 1000 * sum(avg_lats) / num_iters) + " ms.\n")
    f.write("Median latency is: " + str(1000 * avg_lats[int(num_iters / 2)]) + " ms.\n")
    f.write("STD latency is: " + str(1000 * statistics.stdev(avg_lats)) + " ms.\n")

    f.write("Mean exec-time is: " + str(1000 * sum(avg_execs) / num_iters) + " ms.\n")
    f.write("Median exec-time is: " + str(1000 * avg_execs[int(num_iters / 2)]) + " ms.\n")
    f.write("STD exec-time is: " + str(1000 * statistics.stdev(avg_execs)) + " ms.\n")