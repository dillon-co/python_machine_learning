import tensorflow as tf
import numpy as np

all_data = [[],[],[],[]]

for i in all_data:
    for _ in range(500):
        a, b, c, d = np.random.randint(10), np.random.randint(5), np.random.randint(20), np.random.randint(2)
        i.append([a,b,c,d])

train_x, train_y, test_x, test_y = all_data

h1_num_nodes = 50
h2_num_nodes = 50
h3_num_nodes = 20

num_classes = 2


x = tf.placeholder('float', [none, len(train_x[0])])
y = tf.placeholder('float')

def neural_net_model(data):
    hl_1 = {'weights':tf.Variable(tf.random_normal([len(train_x[0], h1_num_nodes)])),
            'biases':tf.Variable(tf.random_normal([h1_num_nodes]))}
    hl_2 = {'weights':tf.Variable(tf.random_normal([h1_num_nodes, h2_num_nodes)])),
            'biases':tf.Variable(tf.random_normal([h2_num_nodes]))}
    hl_3 = {'weights':tf.Variable(tf.random_normal([h2_num_nodes, h3_num_nodes)])),
            'biases':tf.Variable(tf.random_normal([h3_num_nodes]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([h2_num_nodes, h3_num_nodes)])),
                    'biases':tf.Variable(tf.random_normal([h3_num_nodes]))}

    l1 = tf.add(tf.matmul(data, hl_1['weights']), hl_1['biases'])
    l1 = tf.nn.sigmoid(l1)

    l2 = tf.add(tf.matmul(data, hl_1['weights']), hl_1['biases'])
    l2 = tf.nn.sigmoid(l2)

    l3 = tf.add(tf.matmul(data, hl_1['weights']), hl_1['biases'])
    l3 = tf.nn.sigmoid(l3)

    output = tf.matmul(data, hl_1['weights']) + hl_1['biases']

    return output



def train_neural_network(data):
    prediction = neural_network_model(data)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    num_epochs = 50
    with tf.Session() as sess:
        sess.run(global_variables_initializer)
        batch_size = 100
        for epoch in range(num_epochs):
            epoch_loss = 0

            i=0
            while i < train_x:
                start = i
                end = i += batch_size

                x_batch= train_x[start:end]
                y_batch= train_y[start:end]

                _, c = sess.run([optimizer, cost], feed_dict={x: x_batch, y: y_batch})
                epoch_loss += c
                i += batch_size

            print("epoch ", epoch, " of ", num_epochs, ". Loss: -", epoch_loss, "- ")
            correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print("Accuracy = " accuracy.eval({x: test_x, y:test_y}))
