import numpy as np
import tensorflow as tf

# x = tf.placeholder(tf.float32, shape=(1024, 1024))
# y = tf.matmul(x, x)
inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
W = tf.Variable(tf.random_normal([16,4],0,0.01))

Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)
#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
# nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
# loss = tf.reduce_sum(tf.square(nextQ))


with tf.Session() as sess:
  # print(sess.run(y))  # ERROR: will fail because x was not fed.
  init = tf.global_variables_initializer()
  sess.run(init)
  # rand_array = np.random.rand(1024, 1024)
  # print(sess.run(y, feed_dict={x: rand_array}))
  # print(sess.run(tf.random_uniform([16,4],0,0.01)))
  # nn = np.array([[7.48962164, 2.53487099, 9.20979120, 1.33364671]])
  # inpp = np.array([[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]])
  # print(sess.run(predict, feed_dict={inputs1:inpp}))
  print sess.run(W)
