import numpy as np
import tensorflow as tf


num_nodes = 1000
n_classes = 100
n_layers = 3
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128


def deep_net_module(data):
    # This is the default net to be used in all other functions
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x,[-1,chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size, state_is_tuple=False)
    stacked_lstm = rnn.MultiRNNCell([lstm_cell] * n_layers, state_is_tuple=False)

    # outputs_1, states_1 = rnn.static_rnn(lstm_cel, x, dtype=tf.float32)
    outputs, states = rnn.static_rnn(stacked_lstm, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output

def deep_net_module_training():
    #this is for trainingthe individual models

def inputs():
    # This contains all the inputs like a human (sight and sound to start)
    # instead of trying to deal with varying inputs, it's better and easier to
    # have all inputs the same like a human sees

def neural_net_creation():
    # This is the code to create a new neural net when a new thing to learn is
    # found

def tranfer_learning_net():
    # This takes data from all the neural nets in the system and feeds it through
    # a neural net to pretrain the newest model

def goal_finidng_net():
    # This sees a new problem and updates to find the right goal to be achieved
    # (e.g. beating a video game, earning money on the stock market, getting more ad clicks, etc.)

def output_to_output():
    # This uses the outputs of previously trained and functional models as inputs
    # and decides what to do with them (Used in `neural_net_creation`)

def basic_skills_deep_nets():
    # This contains all the neural nets for basic skills
    # (e.g motion & speed detection, face rocognition, etc.)


def metta_net():
    # This uses outputs from basic_deep_nets and transfer learning from
    # transfer_learning_nets to do high level skills (e.g. playing a video game)

def consciousness():
    #this contains all the different parts. ( The whole is greater than the sum )
