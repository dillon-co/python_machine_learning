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
