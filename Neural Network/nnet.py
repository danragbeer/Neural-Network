import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold
import time
import cv2
import os
import matplotlib.pyplot as plt

#letters ds -----------------------------------------------------------------
#Read the csv file and find the size of the file.
train = pd.read_csv('emnist-letters-train.csv', header = None)
test = pd.read_csv('emnist-letters-test.csv', header = None)

train.head()

ds_letters_size = len(train.index)
print("size: ", ds_letters_size)

letters_data = train.iloc[:, 1:]
letters_labels= train.iloc[:, 0]
letters_testdata = test.iloc[:, 1:]
letters_testlabels = test.iloc[:, 0]


letters_labels= pd.get_dummies(letters_labels)
letters_testlabels = pd.get_dummies(letters_testlabels)
letters_labels.head()

letters_data = letters_data.values
letters_labels= letters_labels.values
letters_testdata = letters_testdata.values
letters_testlabels = letters_testlabels.values
del train, test

#-----------------------------------------------------------------------------
n_nodes_hl1 = 5000 #of nodes in hidden layer 1
n_nodes_hl2 = 5000 #of nodes in hidden layer 2

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

batch_size = 100 #size of each batch

n_classes_letters = 26 #number of classes in dataset

def getEpochs():
    '''Ask user for number of epochs'''

    hm_epochs = int(input("How many epochs: "))
    return hm_epochs

def neural_network(data):
    '''This is the neural network architecture that will serve as our prediction'''

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes_letters])),
                    'biases': tf.Variable(tf.random_normal([n_classes_letters]))}


    layer1 = tf.add(tf.matmul(tf.cast(data, tf.float32), hidden_1_layer['weights']), hidden_1_layer['biases'])
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.add(tf.matmul(layer1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    layer2 = tf.nn.relu(layer2)

    output = tf.matmul(layer2, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x, hm_epochs):
    '''Proceeds to train the neural network'''

    prediction = neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction, labels = y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    start_time = time.time()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("Training...")

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range(int(ds_letters_size/batch_size)):
                epx, epy = letters_data[i * batch_size: (i + 1) * batch_size], letters_labels[i * batch_size: (i + 1) * batch_size]
                _, c = sess.run([optimizer, cost], feed_dict = {x: epx, y: epy})
                epoch_loss += c
            print("Epoch", epoch + 1, "loss: ", epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))

        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        print("Testing...")
        print("accuracy: ", accuracy.eval({x:letters_testdata, y:letters_testlabels}))

        print("--- %s seconds ---" % (time.time() - start_time))


def cross_validate(split_size = 4):
    '''Perform cross validation on the neural network'''

    scores = []
    fold = 0
    kf = KFold(n_splits = split_size, shuffle = True)

    with tf.Session() as sess2:
        for train_idx, test_idx in kf.split(letters_data):
            fold += 1
            train_x = letters_data[train_idx]
            train_y = letters_labels[train_idx]
            test_x = letters_data[test_idx]
            test_y = letters_labels[test_idx]

            print("Fold: ", fold)

            start_time = time.time()

            prediction = neural_network(train_x)

            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction, labels = train_y))

            optimizer = tf.train.AdamOptimizer().minimize(cost)

            sess2.run(tf.global_variables_initializer())

            for epoch in range(hm_epochs):

                _, c = sess2.run([optimizer, cost], feed_dict = {x: train_x, y: train_y})
                print("Epoch", epoch + 1)

                print("--- %s seconds ---" % (time.time() - start_time))

                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(train_y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

            scores.append(sess2.run(accuracy, feed_dict = {x: test_x, y: test_y}))

            cv_score = scores

            print("Cross-validation result: ", cv_score)
        print("Test accuracy: ", sess2.run(accuracy, feed_dict={x: letters_testdata, y: letters_testlabels}))


hm_epochs = getEpochs() #get epochs first
train_neural_network(x, hm_epochs) #Train and Test Network
#cross_validate() #Cross validation
