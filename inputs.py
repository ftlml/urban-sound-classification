import tensorflow as tf

def read_and_decode(filename_queue, specgram_shape):

  reader = tf.TFRecordReader()
  # Read an example from the TFRecords file.
  _, example = reader.read(filename_queue)
  features = tf.parse_single_example(example, features={
    'spectrum': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.int64)
  })
  spectrum = tf.decode_raw(features['spectrum'], tf.float32)
  spectrum.set_shape(specgram_shape)
  label = tf.cast(features['label'], tf.int64)
  return spectrum, label

def read_inputs(file_paths, batch_size=64, capacity=1000,
                min_after_dequeue=900, num_threads=2,
                specgram_shape=None):

  with tf.name_scope('input'):
    # Create a file name queue.
    filename_queue = tf.train.string_input_producer(file_paths)
    # Read and decode an example.
    spectrum, label = read_and_decode(filename_queue, specgram_shape)
    # Shuffle the examples and collect them in a queue of batch_size batches.
    spectrums, labels = tf.train.shuffle_batch(
      [spectrum, label], batch_size=batch_size, num_threads=num_threads,
      capacity=capacity, min_after_dequeue=min_after_dequeue
    )
    return spectrums, labels
