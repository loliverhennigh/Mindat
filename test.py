import numpy as np
import tensorflow as tf
import os

import model
import inputs 
import time
import cv2
import loss as ls

FLAGS = tf.app.flags.FLAGS

TRAIN_DIR = "./checkpoints/run_0"

# make list of all mineral
with open('all_minerals.csv', 'r') as f:
  lines = f.readlines()[0]
all_minerals = lines.replace(' ','').split(',')

def print_top_prediction(label, logit, top_k=5):
  max_prediction = dict()
  true_label = []
  for i in xrange(len(all_minerals)):
    max_prediction[all_minerals[i]] = logit[i*2]
    if label[i*2] == 1:
      true_label.append(all_minerals[i])
  max_prediction = sorted(max_prediction.iteritems(), key=lambda (k,v): (-v,k))
  print("True label: " + str(true_label))
  print("Predicted label: " )
  for i in xrange(top_k):
    print(str(max_prediction[i][0]) + ': prob ' + str(max_prediction[i][1]))

def train():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make inputs mineral
    image, label = inputs.inputs_mineral(1, train=False)

    # inference
    logit = model.inference(image) 
    logit_prob = ls.softmax_binary(logit)

    # List of all Variables
    variables = tf.global_variables()

    # Build a saver
    saver = tf.train.Saver(tf.global_variables())   
    #for i, variable in enumerate(variables):
    #  print '----------------------------------------------'
    #  print variable.name[:variable.name.index(':')]

    # Summary op
    summary_op = tf.summary.merge_all()
 
    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session()

    # init if this is the very time training
    sess.run(init)
 
    # init from checkpoint
    saver_restore = tf.train.Saver(variables)
    ckpt = tf.train.get_checkpoint_state(TRAIN_DIR)
    if ckpt is not None:
      print("init from " + TRAIN_DIR)
      saver_restore.restore(sess, ckpt.model_checkpoint_path)

    # Start que runner
    tf.train.start_queue_runners(sess=sess)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.summary.FileWriter(TRAIN_DIR, graph_def=graph_def)

    # calc number of steps left to run
    correct = []
    for step in xrange(1000):
      img, prob_out, label_out = sess.run([image, logit_prob, label])
      print_top_prediction(label_out[0], prob_out[0])
      img = img[0]
      img = img - np.min(img)
      img = 255.0*img/np.max(img)
      img = np.uint8(img)
      img = cv2.resize(img, (330, 330))
      cv2.imshow("image", img)
      cv2.waitKey(0)
      
    

def main(argv=None):  # pylint: disable=unused-argument
  train()

if __name__ == '__main__':
  tf.app.run()
