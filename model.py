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
import nn

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                            """ learning rate """)
tf.app.flags.DEFINE_integer('max_steps',  300000,
                            """ max number of steps to train """)
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """ training batch size """)

# cifar data url
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def inputs_mineral(batch_size, train=True):
  images, labels = inputs.inputs_mineral(batch_size, train)
  return images, labels 

def inference(image, keep_prob=1.0, is_training=True):
  x_i = image
  x_i = nn.conv_layer(x_i, 5, 1, 32, "conv_1", nonlinearity=tf.nn.relu)
  x_i = nn.conv_layer(x_i, 3, 1, 32, "conv_2", nonlinearity=tf.nn.relu)
  x_i = nn.max_pool_layer(x_i, 2, 2)
  x_i = nn.conv_layer(x_i, 3, 1, 64, "conv_3", nonlinearity=tf.nn.relu)
  x_i = nn.conv_layer(x_i, 3, 1, 64, "conv_4", nonlinearity=tf.nn.relu)
  x_i = nn.max_pool_layer(x_i, 2, 2)
  x_i = nn.conv_layer(x_i, 3, 1, 128, "conv_5", nonlinearity=tf.nn.relu)
  x_i = nn.conv_layer(x_i, 3, 1, 128, "conv_6", nonlinearity=tf.nn.relu)
  x_i = nn.max_pool_layer(x_i, 2, 2)
  x_i = nn.conv_layer(x_i, 3, 1, 256, "conv_7", nonlinearity=tf.nn.relu)
  x_i = nn.conv_layer(x_i, 3, 1, 256, "conv_8", nonlinearity=tf.nn.relu)
  x_i = nn.max_pool_layer(x_i, 2, 2)
  x_i = nn.fc_layer(x_i, 1028, "fc_0", flat=True, nonlinearity=tf.nn.relu) 
  x_i = tf.nn.dropout(x_i, keep_prob)
  x_i = nn.fc_layer(x_i, 512, "fc_2", nonlinearity=tf.nn.relu) 
  x_i = tf.nn.dropout(x_i, keep_prob)
  label = nn.fc_layer(x_i, 2*140, "fc_classes", nonlinearity=None) 
  #label = inception_model.inception_v3(image, dropout_keep_prob=keep_prob, num_classes=140*2, is_training=is_training)
  return label

def loss(label, logits):
  loss = ls.cross_entropy_binary(label, logits)
  tf.summary.scalar('loss', loss)
  return loss

def train(total_loss, lr, global_step):
   train_op = tf.train.AdamOptimizer(lr).minimize(total_loss, global_step)
   return train_op

