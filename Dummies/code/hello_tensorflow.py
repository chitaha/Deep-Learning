"""A simple TensorFlow application"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

# create tensor
msg = tf.string_join(["Hello ", "TensorFlow!"])

# launch session
with tf.Session() as sess:
    print(sess.run(msg))
