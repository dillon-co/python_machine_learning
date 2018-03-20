import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from reformated_earthquake_data import create_featureset_and_labels


train_x ,train_y, test_x, test_y = create_featureset_and_labels()
print(train_x[0])
print(train_y[0])
# print(("; ").join(train_y))
# print(train_x)


batch_size = 100
hm_epochs = 2
n_layers = 2
n_classes = 2
chunk_size = 3
n_chunks = 28
rnn_size = 100


x = tf.placeholder('float', [100, 3])
y = tf.placeholder('float', [None, n_classes])

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    # print(x.get_shape())
    # x = tf.transpose(x)
    # x = tf.reshape(x,[-1,chunk_size])
    # x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size, state_is_tuple=False)
    # stacked_lstm = rnn.MultiRNNCell([lstm_cell] * n_layers, state_is_tuple=False)

    # print(stacked_lstm.get_shape())
    # outputs_1, states_1 = rnn.static_rnn(lstm_cel, x, dtype=tf.float32)
    outputs, states = rnn.static_rnn(lstm_cell, [x], dtype=tf.float32)
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
    return output


def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    num_epochs = 50
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            epoch_loss = 0

            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = np.array(train_x[start:end], dtype=object)
                batch_y = np.array(train_y[start:end], dtype=object)
                # print(batch_x)
                # print("ITR ", i)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch+1, 'completed out of', num_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print("lol")
        print('Accuracy:', accuracy.eval({x:test_x[0:100], y:test_y[0:100]}))


train_neural_network(x)
