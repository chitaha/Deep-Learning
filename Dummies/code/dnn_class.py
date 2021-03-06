"""
The dnn_class.py module demonstrates how to create a DNNClassifier
and use it to classify data points. It loads MNIST training data from
mnist_train. tfrecords and loads test data from mnist_test.tfrecords
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

# Constants
image_dim = 28
num_labels = 10
batch_size = 80
num_steps = 8000
hidden_layers = [128, 32]


# Function to parse MNIST TFRecords
def parser(record):
    features = tf.parse_single_example(record, features={'images': tf.FixedLenFeature(
        [], tf.string), 'labels': tf.FixedLenFeature([], tf.int64), })
    image = tf.decode_raw(features['images'], tf.uint8)
    image.set_shape([image_dim * image_dim])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = features['labels']
    return image, label


# Create the DNNClassifier
column = tf.feature_column.numeric_column(
    'pixels', shape=[image_dim * image_dim])
dnn_class = tf.estimator.DNNClassifier(
    hidden_layers, [column], model_dir='dnn_output', n_classes=num_labels)


# Train the estimator
def train_func():
    dataset = tf.data.TFRecordDataset('mnist_train.tfrecords')
    dataset = dataset.map(parser).repeat().batch(batch_size)
    image, label = dataset.make_one_shot_iterator().get_next()
    return {'pixels': image}, label

dnn_class.train(train_func, steps=num_steps)


# Test the estimator
def test_func():
    dataset = tf.data.TFRecordDataset('mnist_test.tfrecords')
    dataset = dataset.map(parser).batch(batch_size)
    image, label = dataset.make_one_shot_iterator().get_next()
    return {'pixels': image}, label

metrics = dnn_class.evaluate(test_func)

# Display metrics
print('\nEvaluation metrics:')
for key, value in metrics.items():
    print(key, ': ', value)
