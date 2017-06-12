
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

def int_shape(x):
  return list(map(int, x.get_shape()))

def cross_entropy_binary(label, logit):
  label_shape = int_shape(label)
  # reshape label vectore
  label = tf.reshape(label, [label_shape[0]*label_shape[1]/2, 2])
  # reshape logits vector
  logit = tf.reshape(logit, [label_shape[0]*label_shape[1]/2, 2])
  # loss
  loss = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit)
  loss = tf.reduce_mean(loss) 
  return loss, label


