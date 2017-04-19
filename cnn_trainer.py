import os
import numpy as np
import tensorflow as tf
import time

from constants import CLASSES, MAX_SAMPLES, SOUND_FILE_DIRS, \
                      SOUND_FILE_PATHS, SPECGRAM_SHAPE, TF_RECORDS_META, \
                      TF_RECORDS_DESTINATION, TF_RECORDS_TEST, TF_RECORDS_TRAIN
from inputs import read_inputs
from preprocessing import prepare_data

INPUT_PIPELINE_THREADS = 4

BATCH_SIZE = 1
MINI_BATCHES = 9
EPOCHS = 5000
LEARNING_RATE = 1e-3

N_CELLS = 250
N_CLASSES = len(CLASSES)

print('Dataset Size: {}\n'.format(len(SOUND_FILE_PATHS)))

# If the data hasn't been preprocessed, then do it now.
if not os.path.exists(TF_RECORDS_META) and \
   not os.path.exists(TF_RECORDS_TRAIN) and \
   not os.path.exists(TF_RECORDS_TEST):
  FEATURES_MIN, FEATURES_MAX, FEATURES_MEAN = prepare_data(
    SOUND_FILE_PATHS, TF_RECORDS_DESTINATION, MAX_SAMPLES
  )
  with open(TF_RECORDS_META, 'w') as OUTPUT:
    OUTPUT.write('{},{},{}'.format(FEATURES_MIN, FEATURES_MAX, FEATURES_MEAN))
else:
  with open(TF_RECORDS_META, 'r') as INPUT:
    META_DATA = INPUT.readline()
    FEATURES_MIN, FEATURES_MAX, FEATURES_MEAN = [
      float(DATA_POINT) for DATA_POINT in META_DATA.split(',')
    ]

print('Training Set Size: {}'.format(int(len(SOUND_FILE_PATHS) * .9)))
print('Test Set Size: {}\n'.format(int(len(SOUND_FILE_PATHS) * .1)))

def variable_on_cpu(name, shape, initializer, dtype=tf.float32):
  '''
  Create a shareable variable.
  '''

  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def cnn_layer(X, shape, strides, scope, padding='SAME'):
  '''
  Create a convolution layer.
  '''

  kernel = variable_on_cpu(
    'kernel', shape, tf.contrib.layers.xavier_initializer(dtype=tf.float32)
  )
  conv = tf.nn.conv2d(X, kernel, strides, padding=padding)
  biases = variable_on_cpu('b', [shape[-1]], tf.constant_initializer(0.0))
  activation = tf.nn.relu(conv + biases, name=scope.name)
  tf.summary.histogram('{}/activations'.format(scope.name), activation)
  tf.summary.scalar(
    '{}/sparsity'.format(scope.name), tf.nn.zero_fraction(activation)
  )
  return activation

def mlp_layer(X, n_in, n_out, scope, act_func=None):
  '''
  Create a feedforward layer.
  '''

  weights = variable_on_cpu(
    'W', [n_in, n_out], tf.contrib.layers.xavier_initializer(dtype=tf.float32)
  )
  biases = variable_on_cpu('b', [n_out], tf.constant_initializer(0.0))
  if act_func is not None:
    activation = act_func(tf.matmul(X, weights) + biases, name=scope.name)
  else:
    activation = tf.add(tf.matmul(X, weights), biases, name=scope.name)
  tf.summary.histogram('{}/activations'.format(scope.name), activation)
  tf.summary.scalar(
    '{}/sparsity'.format(scope.name), tf.nn.zero_fraction(activation)
  )
  return activation

def pool_layer(X, window_size, strides, scope, padding='SAME'):
  '''
  Create a max pooling layer.
  '''

  return tf.nn.max_pool(X, ksize=window_size, strides=strides,
                        padding=padding, name=scope.name)

# Build our dataflow graph.
GRAPH = tf.Graph()
with GRAPH.as_default():
  SPECTRUMS, LABELS = read_inputs(
    [TF_RECORDS_TRAIN], specgram_shape=[np.prod(SPECGRAM_SHAPE),],
    batch_size=BATCH_SIZE, min_after_dequeue=BATCH_SIZE,
    num_threads=INPUT_PIPELINE_THREADS
  )
  # Remove the mean and scale between -1 and 1.
  SPECTRUMS = (SPECTRUMS - FEATURES_MEAN) / (FEATURES_MAX - FEATURES_MIN)
  # Reshape the inputs for consumption by the convolution layers.
  SPECTRUMS = tf.reshape(
    SPECTRUMS, [BATCH_SIZE, SPECGRAM_SHAPE[0], SPECGRAM_SHAPE[1]]
  )
  SPECTRUMS = tf.expand_dims(SPECTRUMS, -1)
  # Build two convolution layers followed by max pooling.
  with tf.variable_scope('layer1') as scope:
    H_1 = cnn_layer(SPECTRUMS, (5, 5, 1, 64), (1, 1, 1, 1), scope)
  with tf.variable_scope('layer2')as scope:
    H_2 = pool_layer(H_1, (1, 3, 3, 1), (1, 2, 2, 1), scope)
  with tf.variable_scope('layer3') as scope:
    H_3 = cnn_layer(H_2, (5, 5, 64, 64), (1, 1, 1, 1), scope)
  with tf.variable_scope('layer4') as scope:
    H_4 = pool_layer(H_3, (1, 3, 3, 1), (1, 2, 2, 1), scope)
  # Reshape the inputs for consumption by the final feed forward and classifier
  # classifier layers.
  H_4 = tf.reshape(H_4, (BATCH_SIZE, -1))
  # Build two feedforward layers.
  with tf.variable_scope('layer5') as scope:
    H_5 = mlp_layer(
      H_4, H_4.get_shape()[1].value, N_CELLS, scope, act_func=tf.nn.relu,
    )
  with tf.variable_scope('layer6') as scope:
    H_6 = mlp_layer(H_5, N_CELLS, N_CELLS, scope, act_func=tf.nn.relu)
  # Build the classifier layer.
  with tf.variable_scope('layer7') as scope:
    H_7 = mlp_layer(H_6, N_CELLS, N_CLASSES, scope)
  Y = tf.nn.softmax(H_7)
  # Compute the cross entropy loss.
  COST = tf.nn.sparse_softmax_cross_entropy_with_logits( 
    logits=H_7, labels=LABELS
  )
  COST = tf.reduce_mean(COST)
  tf.summary.scalar("cost", COST)
  # Compute gradients.
  OPTIMIZER = tf.train.AdamOptimizer(LEARNING_RATE)
  GRADIENTS = OPTIMIZER.compute_gradients(COST)
  # Apply gradients.
  APPLY_GRADIENT_OP = OPTIMIZER.apply_gradients(GRADIENTS)
  # Add histograms for gradients.
  for GRADIENT, VAR in GRADIENTS:
    if GRADIENT is not None:
      tf.summary.histogram('{}/gradients'.format(VAR.op.name), GRADIENT)
  # Collect the TensorBoard summaries.
  SUMMARIES_OP = tf.summary.merge_all()

# Start training the model.
with tf.Session(graph=GRAPH) as SESSION:
  COORDINATOR = tf.train.Coordinator()
  THREADS = tf.train.start_queue_runners(SESSION, COORDINATOR)
  # Initialize all the variables.
  SESSION.run(tf.global_variables_initializer())
  # Create a tensorflow summary writer.
  SUMMARY_WRITER = tf.summary.FileWriter('Tensorboard/CNN', graph=GRAPH)
  # Create a tensorflow graph writer.
  GRAPH_WRITER = tf.train.Saver()

  TOTAL_DURATION = 0.0
  for EPOCH in range(EPOCHS):
    DURATION = 0
    ERROR = 0.0
    START_TIME = time.time()
    for MINI_BATCH in range(MINI_BATCHES):
      _, SUMMARIES, COST_VAL, PREDICTION = SESSION.run([
        APPLY_GRADIENT_OP, SUMMARIES_OP, COST, Y
      ])
      ERROR += COST_VAL
    DURATION += time.time() - START_TIME
    TOTAL_DURATION += DURATION
    # Write the summaries to disk.
    SUMMARY_WRITER.add_summary(SUMMARIES, EPOCH)
    # Update the console.
    print('Epoch %d: loss = %.2f (%.3f sec)' % (EPOCH, ERROR, DURATION)) 
    if EPOCH == EPOCHS - 1 or ERROR < 0.005:
      print(
        'Done training for %d epochs. (%.3f sec)' % (EPOCH, TOTAL_DURATION)
      )
      break
  GRAPH_WRITER.save(SESSION, 'Data/urban_sound_8k.model')
  COORDINATOR.request_stop()
  COORDINATOR.join(THREADS)
