import os

from audioread import NoBackendError

import librosa
import numpy as np
import tensorflow as tf

from python_speech_features.sigproc import framesig, logpowspec

def bytes_feature(value):
  '''
  Creates a TensorFlow Record Feature with value as a byte array.
  '''

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
  '''
  Creates a TensorFlow Record Feature with value as a 64 bit integer.
  '''

  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def prepare_data(example_paths, destination_dir, max_len, nfft=256,
                 noverlap=128, test_portion=0.1):
  '''
  Partitions the data into two sets training and test and saves each set
  into two separate files in a TensorFlow native format.

  NOTE: This can be refactored into two separate methods.

  :return A tuple container (min_val, max_val. mean) for the entire
          dataset.
  '''

  # Compute the split point between the training and test sets.
  split_point = int(len(example_paths) * (1 - test_portion))
  # Write the training set.
  train_set_path = os.path.join(destination_dir, 'urban_sound_train.tfrecords')
  results = prepare_tfrecord(
    example_paths[:split_point], train_set_path, max_len, nfft=nfft,
    noverlap=noverlap
  )
  # Write the test set.
  test_set_path = os.path.join(destination_dir, 'urban_sound_test.tfrecords')
  prepare_tfrecord(
    example_paths[split_point:], test_set_path, max_len, nfft=nfft,
    noverlap=noverlap
  )
  return results


def prepare_tfrecord(example_paths, destination_path, max_len, nfft=256,
                     noverlap=128):
  '''
  Converts a set of inputs into spectrograms and saves them to disk in
  a TensorFlow native format.

  :return A tuple container (min_val, max_val. mean) for the set.
  '''

  features_min, features_max = None, None
  features_count, features_sum = 0, 0
  # Open a TFRecords file for writing.
  writer = tf.python_io.TFRecordWriter(destination_path)
  for idx in range(len(example_paths)):
    # Load an audio file for preprocessing.
    try:
      samples, _ = librosa.load(example_paths[idx])
    except NoBackendError:
      print('Warning: Could not load {}.'.format(example_paths[idx]))
      continue
    # Pad or shorten the number audio samples to max length.
    if samples.shape[0] < max_len:
      samples = np.pad(samples, (0, max_len - samples.shape[0]), 'constant')
    elif samples.shape[0] > max_len:
      samples = samples[:max_len]
    # Generate a log power spectrum of the audio samples.
    spectrum = np.abs(logpowspec(
      framesig(samples, nfft, noverlap, winfunc=np.hanning), nfft, norm=0
    ))
    spectrum = np.transpose(np.flip(spectrum, 1)).astype(np.float32)
    label = int(os.path.split(example_paths[idx])[-1].split('-')[1])
    # Keep track of the dataset statistics.
    new_min = np.min(spectrum)
    new_max = np.max(spectrum)
    if features_min is not None and features_max is not None:
      features_min = new_min if features_min > new_min else features_min
      features_max = new_max if features_max < new_max else features_max
    else:
      features_min = new_min
      features_max = new_max
    features_count += np.prod(spectrum.shape)
    features_sum += np.sum(spectrum)
    # Write the final spectrum and label to disk.
    example = tf.train.Example(features=tf.train.Features(feature={
      'spectrum': bytes_feature(spectrum.flatten().tostring()),
      'label': int64_feature(label)
    }))
    writer.write(example.SerializeToString())
  writer.close()
  # Return the dataset statistics.
  return features_min, features_max, float(features_sum) / features_count
