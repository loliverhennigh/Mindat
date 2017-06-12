"""Builds the ring network.
Summary of available functions:
  # Compute pics of the simulation runnig.
  
  # Create a graph to train on.
"""


import tensorflow as tf
import numpy as np
import inputs
import loss as ls
import inception_model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                            """ learning rate """)
tf.app.flags.DEFINE_integer('max_steps',  300000,
                            """ max number of steps to train """)
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """ training batch size """)

# cifar data url
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def inputs_mineral(batch_size, train=True):
  images, labels = inputs.inputs_mineral(batch_size, train)
  return images, labels 

def inference(image, keep_prob=1.0, is_training=True):
  label = inception_model.inception_v3(image, dropout_keep_prob=keep_prob, num_classes=140*2, is_training=is_training)
  return label

def loss(label, logits):
  loss, reshaped_label = ls.cross_entropy_binary(label, logits)
  tf.summary.scalar('loss', loss)
  return loss, reshaped_label

def train(total_loss, lr, global_step):
   train_op = tf.train.AdamOptimizer(lr).minimize(total_loss, global_step)
   return train_op

