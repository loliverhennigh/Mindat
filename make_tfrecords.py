
import os
import sys
import imghdr
import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm

def load_image(image_name, shape = [299,299]):
  image = cv2.imread(image_name)
  if image is None:
    return image
  # pad top and bottom of image
  pad_top_length = image.shape[0] - image.shape[1]
  if pad_top_length > 0:
    image = np.concatenate([np.zeros((image.shape[0], pad_top_length - int(pad_top_length/2), image.shape[2])), image, np.zeros((image.shape[0], int(pad_top_length/2), image.shape[2]))], axis=1)
  pad_left_length = image.shape[1] - image.shape[0]
  if pad_left_length > 0:
    image = np.concatenate([np.zeros((pad_left_length - int(pad_left_length/2), image.shape[1], image.shape[2])), image, np.zeros((int(pad_left_length/2), image.shape[1], image.shape[2]))], axis=0)
  image = np.uint8(image)
  image = cv2.resize(image, (shape[0], shape[1]))
  return image 

def mineral_name_vector(minerals, all_minerals):
  mineral_vec = np.zeros(len(all_minerals))
  for i in xrange(len(all_minerals)):
    if all_minerals[i] in minerals:
      mineral_vec[i] = 1.0
  mineral_vec = np.uint8(mineral_vec)
  return mineral_vec

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# make list of images and mineral names
with open('img_url_list_converted.csv', 'r') as f:
  lines = f.readlines()
url_list = []
minerals_list = []
for l in lines:
  url_list.append(l.split(',')[0]) 
  minerals_list.append(l.replace(' ','').split(',')[1:-1])

# make list of all mineral
with open('all_minerals.csv', 'r') as f:
  lines = f.readlines()[0]
all_minerals = lines.replace(' ','').split(',')[:-1]
print(len(all_minerals))

base_dir = './tfrecords/'
train_mineral_writers = dict()
test_mineral_writers = dict()
for m in all_minerals:
  train_record_filename = base_dir + m + '_train.tfrecord'
  train_mineral_writers[m] = tf.python_io.TFRecordWriter(train_record_filename)
  test_record_filename = base_dir + m + '_test.tfrecord'
  test_mineral_writers[m] = tf.python_io.TFRecordWriter(test_record_filename)

for i in tqdm(xrange(len(url_list))):
  image_name = '/data/mindat-images/' + '_'.join(url_list[i].split('/')[3:])
  image = load_image(image_name)
  if image is None:
    continue
  image = image.tostring()
  label = mineral_name_vector(minerals_list[i], all_minerals).tostring()
  example = tf.train.Example(features=tf.train.Features(feature={
    'label': _bytes_feature(label),
    'image': _bytes_feature(image)}))
  split = np.random.rand()
  if split > .8:
    test_mineral_writers[minerals_list[i][0]].write(example.SerializeToString())
  else:
    train_mineral_writers[minerals_list[i][0]].write(example.SerializeToString())
 


