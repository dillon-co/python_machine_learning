import tflearn, sys
import numpy as np
import tensorflow as tf
input_text = open('pos.txt').read().lower()
chars = sorted(list(set(input_text)))
chatint = dict((char, ints) for ints, char in enumerate(chars))
intchar = dict((ints, char) for ints, char in enumerate(chars))

filename = 'mine'
seqlen = 100
lstmhid = 320
keeprate = 0.80
train = []
true = []
tf.reset_default_graph()
for i in range(0, len(input)-seqlen, 1)
    train.append([charint[char] for char in input_text[i:i+seqlen]])
    true.append([charint[input_text[i:i+seqlen]])






X = tf.placeholder()


def recurrent_neural_network(x):
    
    # print("first x is: %s", x)
    # x = tf.transpose(x, [1,0,2])
    # # print("second x is: %s", x)
    # x = tf.reshape(x,[-1,chunk_size])
    # # print("new x is: %s", x)
    # x = tf.split(x, n_chunks, 0)
    # print("final x is:", x)

    lstm_cell = rnn.BasicLSTMCell(rnn_size, state_is_tuple=True)
    # stacked_lstm = rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(n_layers)], state_is_tuple=True)
    # outputs, states = rnn.static_rnn(stacked_lstm, x, dtype=tf.float32)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output

def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x:mnist.test.images.reshape((-1,n_chunks,chunk_size)), y:mnist.test.labels}))



train_neural_network(x)
