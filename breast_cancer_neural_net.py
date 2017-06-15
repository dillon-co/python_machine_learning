import numpy as np
import pandas as pd
import tensorflow as tf

## Import Data
df = pd.read_csv('breast-cancer-wisconsin.data.txt', error_bad_lines=False)
# print np.array(df)[20]
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

### Create Meta Data
num_imputs = len(x[0])
layer_1_nodes = 30
layer_2_nodes = 30

num_classes = 2

batch_size = 100

inputs = tf.placeholder(shape=[1],dtype=tf.float32)
expected_outputs = tf.polaceholder(shape=[1],dtype=tf.float32)

### Create our weights
hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([num_imputs, layer_1_nodes])),
                  'biases':tf.Variable(tf.random_normal([layer_1_nodes]))}

hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([layer_1_nodes, layer_2_nodes])),
                  'biases':tf.Variable(tf.random_normal([layer_2_nodes]))}

output_layer = {'weights':tf.Variable(tf.random_normal([layer_2_nodes, num_classes])),
                  'biases':tf.Variable(tf.random_normal([num_classes]))}


### Run Optimization
def neural_network(data):
    layer_1 = tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1,hidden_layer_2['weights']),hidden_layer_2['biases'])
    layer_2 = tf.nn.relu(layer_2)

    output = tf.matmul(layer_2, output_layer['biases']) + output_layer['weights']

    return output

### Start Session and initialize variables
def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.cross_entropy_with_logits(logits=prediction, labels=y))################## Look Up Function #############################
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    num_episodes = 500

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        ses.run(init)

        ### Loop through data and run tests
        for episode in range(num_episodes):
            episode_loss = 0

            i = 0
            while i < len(X_train):
                start = i
                end = i + batch_sixe

                batch_x = np.array(X_train[start:end])
                batch_y = np.array(Y_train[start:end])

                _, c = sess.run([optimizer, cost],feed_dict={x:batch_x,y:batch_y})
                episode_loss += c
                i += batch_size
                ### Calculate Accuracy
            print("Episode", episode+1, "completed out of ", num_episodes, "Loss: ", episode_loss)
            correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
            print("Accuracy", accuracy.eval({text_x:x,tesy_y:y}))################################### Look Up Function ####################################


train_neural_network(x)
