from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

# Add two scalars
a = tf.constant(2.5)
b = tf.constant(4.5)
total = a + b

# Create operations that generate summary data
tf.summary.scalar("a", a)
tf.summary.scalar("b", b)
tf.summary.scalar("total", total)

# Merge the operations into a single operation
merged_op = tf.summary.merge_all()
with tf.Session() as sess:
_, summary = sess.run([sum, merged_op])
