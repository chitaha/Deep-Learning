"""
The combined.py module analyzes census data.
To be precise, the module creates a DNNLinearCombined
Classifier that examines data from the 1994 Census to
determine whether a person will make more or less than
$50,000 per year.
I provide the census data in two CSV files: adult.data contains training
data and adult.test contains test data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import pandas as pd

# Define column headings
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
           'marital_status', 'occupation', 'relationship', 'race', 'gender',
           'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
           'income_bracket']

# Create feature columns
age = tf.feature_column.numeric_column('age')
workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    'workclass', ['Private', 'Self-emp-not-inc', 'self-emp-inc', 'Federal-gov',
                  'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
fnlwgt = tf.feature_column.numeric_column('fnlwgt')
education = tf.feature_column.categorical_column_with_vocabulary_list(
    'education', [...])
education_num = tf.feature_column.numeric_column('education_num')
marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    'marital_status', ['Never-married', 'Divorced', 'Separated', 'Widowed',
                       'Married-civ-spouse', 'Married-AF-spouse',
                       'Married-spouse-absent'])
occupation = tf.feature_column.categorical_column_with_vocabulary_list(
    'occupation', [...])
relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    'relationship', ['Wife', 'Own-child', 'Husband', 'Not-in-family',
                     'Other-relative', 'Unmarried'])
race = tf.feature_column.categorical_column_with_vocabulary_list(
    'race', ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
             'Other', 'Black'])
gender = tf.feature_column.categorical_column_with_vocabulary_list(
    'gender', ['Female', 'Male'])
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')
native_country = tf.feature_column.categorical_column_with_vocabulary_list(
    'native_country', [...])

# Create groups of columns
linear_columns = [
    tf.feature_column.crossed_column(
        ['education', 'occupation'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        ['native_country', 'occupation'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        ['workclass', 'occupation'], hash_bucket_size=1000)]

dnn_columns = [
    tf.feature_column.indicator_column(workclass),
    tf.feature_column.indicator_column(education),
    tf.feature_column.indicator_column(gender),
    tf.feature_column.indicator_column(relationship),
    tf.feature_column.indicator_column(native_country),
    tf.feature_column.indicator_column(occupation),
    age, education_num, capital_gain, capital_loss,
    hours_per_week, fnlwgt]

# Create classifier
classifier = tf.estimator.DNNLinearCombinedClassifier(
    linear_feature_columns=linear_columns,
    dnn_feature_columns=dnn_columns, dnn_hidden_units=[120, 60])

# Train the classifier
train_file = open('adult.data', 'r')
train_frame = pd.read_csv(train_file,
                          names=columns, engine='python',
                          skipinitialspace=True, skiprows=1)
train_labels = train_frame['income_bracket'].apply(lambda x: '>50K' in x)
train_fn = tf.estimator.inputs.pandas_input_fn(
    x=train_frame, y=train_labels,
    batch_size=100, num_epochs=600,
    shuffle=True)
classifier.train(train_fn)

# Test the estimator
test_file = open('adult.test', 'r')
test_frame = pd.read_csv(test_file,
                         names=columns, engine='python',
                         skipinitialspace=True, skiprows=1)
test_labels = test_frame['income_bracket'].apply(lambda x: '>50K' in x)
test_fn = tf.estimator.inputs.pandas_input_fn(
    x=test_frame, y=test_labels,
    num_epochs=1, shuffle=False)
metrics = classifier.evaluate(test_fn)

# Display metrics
print('\nEvaluation metrics:')
for key, value in metrics.items():
    print(key, ': ', value)
