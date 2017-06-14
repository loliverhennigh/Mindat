
import tensorflow as tf
import numpy as np

from minerals_ratios import mineral_ratios

FLAGS = tf.app.flags.FLAGS

def int_shape(x):
  return list(map(int, x.get_shape()))

def cross_entropy_binary(label, logit):
  label = label[:,:-2]
  logit = logit[:,:-2]
  """
  label = label[:,8:10]
  logit = logit[:,8:10]
  loss = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit)
  ratio = label[:,0]*(1.0/0.01936368421892) + 1.0
  loss = loss*ratio
  loss = tf.reduce_mean(loss) 
  """
  logit_orig = logit
  # get shape
  label_shape = int_shape(label)
  # reshape label vectore
  label = tf.reshape(label, [label_shape[0]*label_shape[1]/2, 2])
  # reshape logits vector
  logit = tf.reshape(logit, [label_shape[0]*label_shape[1]/2, 2])
  # loss
  loss = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit)
  loss = tf.reshape(loss, [label_shape[0], label_shape[1]/2])
  min_ratios = tf.constant(mineral_ratios) 
  min_ratios = 1.0/tf.reshape(min_ratios, [1, label_shape[1]/2])
  label_ratios = tf.reshape(label[:,0], [label_shape[0], label_shape[1]/2])
  ratios = ((min_ratios-1.0) * label_ratios) + 1.0
  loss_norm = loss * ratios
  # probability loss 
  #logit_prob = tf.reshape(logit, [label_shape[0], label_shape[1]/2, 2])[:,:,0]
  #loss_prob = tf.reduce_sum(
  #loss = tf.reduce_sum(loss_norm) 
  loss = tf.reduce_mean(loss_norm) 
  #loss = tf.nn.l2_loss(label-logit)
  return loss

def softmax_binary(inputs):
  inputs_shape = int_shape(inputs)
  inputs = tf.reshape(inputs, [inputs_shape[0]*inputs_shape[1]/2, 2])
  inputs = tf.nn.softmax(inputs)
  inputs = tf.reshape(inputs, [inputs_shape[0], inputs_shape[1]])
  return inputs

