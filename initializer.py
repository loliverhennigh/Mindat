
import tensorflow as tf

def initialize_inception_v3(sess, variables):
  # weight variables 
  newVars = variables[385:]
  oldVars = variables[1:385]
  ind = 0
  #for var in oldVars:
  for var in variables:
    ind += 1
    print(var.name)
    print(ind)
  tmpSaver = tf.train.Saver(oldVars)

  # make initalizers
  init_last = tf.initialize_variables(newVars)
  init_global_step = tf.initialize_variables([variables[0]])
  
  sess.run(init_last)
  sess.run(init_global_step)
  tmpSaver.restore(sess, './weights/inception_v3.ckpt')
