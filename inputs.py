
import os
import numpy as np
import tensorflow as tf
from glob import glob as glb

FLAGS = tf.app.flags.FLAGS

def read_data_mineral(filename_queue):
  reader = tf.TFRecordReader()
  key, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
      'label':tf.FixedLenFeature([],tf.string),
      'image':tf.FixedLenFeature([],tf.string),
    }) 
  label = tf.decode_raw(features['label'], tf.uint8)
  image = tf.decode_raw(features['image'], tf.uint8)
  label = tf.reshape(label, [122])
  image = tf.reshape(image, [299, 299, 3])
  label = tf.to_float(label)
  image = tf.to_float(image)
  return image, label 

def _generate_image_label_batch_mineral(image, label, batch_size):
  num_preprocess_threads = 1
  #Create a queue that shuffles the examples, and then
  images, labels = tf.train.shuffle_batch(
    [image, label],
    batch_size=batch_size,
    num_threads=num_preprocess_threads,
    capacity=1000 + 3 * batch_size,
    min_after_dequeue=1000)
  return images, labels

def inputs_mineral(batch_size, train=True):
  if train:
    tfrecord_filename = glb('./tfrecords/*train.tfrecord') 
  else:
    tfrecord_filename = glb('./tfrecords/*test.tfrecord') 
  filename_queue = tf.train.string_input_producer(tfrecord_filename) 

  image, label = read_data_mineral(filename_queue)

  # data augmentation
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_brightness(image, max_delta=15)
  image = tf.image.random_contrast(image, 0.9, 1.1)
  image = tf.random_crop(image, [250, 250, 3])
  image = tf.reshape(image, [1, 250, 250, 3])
  image = tf.image.resize_bicubic(image, [299, 299])
  image = tf.reshape(image, [299, 299, 3])
  #image = tf.image.per_image_standardization(image)
    
  # display in tf summary page 
  images, labels  = _generate_image_label_batch_mineral(image, label, batch_size)
  tf.summary.image('mineral image', images)
  return images, labels

