import time
from numpy.lib.type_check import imag
import tensorflow as tf
import numpy as np

def load_pb(path_to_pb):
    with tf.io.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

test_images = np.random.randint(low =0, high= 256, size = [32, 224, 224, 3], dtype=np.uint8)
test_images = test_images / 255.0
#test_images = tf.cast(test_images, dtype='float32')

graph = load_pb('/home/fqararyah/MobileNetV3Small.pb')

#print(graph.get_operations())
input = graph.get_tensor_by_name('input_1:0')
output = graph.get_tensor_by_name('Predictions/Softmax:0')

sess = tf.compat.v1.Session(graph=graph)

t0 = time.time()
for i in range(0, 10):
	sess.run(output, feed_dict={input: test_images})
print(time.time() - t0)
