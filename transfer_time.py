from tensorflow.python import util
import BenchmarkModel as bm
import tensorflow as tf
import time

bm.load_images()
test_images = bm.test_images

tf.compat.v1.disable_eager_execution() # need to disable eager in TF2.x
# Build a graph.
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * test_images

# Launch the graph in a session.
sess = tf.compat.v1.Session()

# Evaluate the tensor `c`.
t0 = time.time()
sess.run(c)
print(time.time() - t0)

for i in range(0, 1000):
    c = c * a

# Evaluate the tensor `c`.
t0 = time.time()
res = sess.run(c)
print(time.time() - t0)
print(res.shape)