import numpy as np
import tensorflow as tf


from create_sentiment_featuresets import create_feature_sets_and_labels

train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')


n_nodes_h11 = 500
n_nodes_h12 = 500
n_nodes_h13 = 500

all_nodes = [len(train_x[0]), n_nodes_h11, n_nodes_h12, n_nodes_h13]

n_classes = 2
batch_size = 100

x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')

def neural_net_layer(x, h, c, weights):
    #forget gate layer
    f_gate_bias = tf.Variable(tf.random_normal)
    cell_state = tf.placeholder()
    ft = tf.nn.sigmoid(tf.add(tf.matmul(weights['weights'], [h, x]), weights['biases']))
    a =

def neural_network_model(data, h_t, ):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_h11])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_h11]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_h11, n_nodes_h12])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_h12]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_h12, n_nodes_h13])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_h13]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_h13, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}

    # l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    # l1 = tf.nn.relu(l1)
    #
    # l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    # l2 = tf.nn.relu(l2)
    #
    # l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    # l3 = tf.nn.relu(l3)
    #
    # output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output
