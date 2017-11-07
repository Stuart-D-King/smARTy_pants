'''
Code adapted from Ankit Sachan's Tensorflow Tutorial 2: image classifier using convolutional neural network blog post (http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/) and GitHub repository (https://github.com/sankit1/cv-tricks.com/tree/master/Tensorflow-tutorials/tutorial-2-image-classifier)
'''

import numpy as np
from numpy.random import seed
import tensorflow as tf
from tensorflow import set_random_seed
from random import sample
import pickle, pdb
from keras.utils import np_utils

'''
Train and save a CNN model with three convolutional layers using TensorFlow
'''

seed(1337)
set_random_seed(2)

# prepare input data
with open('data/class_dict.pkl', 'rb') as f:
    class_dict = pickle.load(f)

classes = class_dict.values()
num_classes = len(classes)

img_size = 299
num_channels = 3

with open('data/train_data_299.pkl', 'rb') as f:
    train_data = pickle.load(f)

images = [img for img, label in train_data]
X_train = np.stack(images)
y_train = np.array([label for img, label in train_data], dtype=np.int32)
y_train = np_utils.to_categorical(y_train, num_classes)

train_data = list(zip(X_train, y_train))

with open('data/test_data_299.pkl', 'rb') as f:
    test_data = pickle.load(f)

images = [img for img, label in test_data]
X_test = np.stack(images)
y_test = np.array([label for img, label in test_data], dtype=np.int32)
y_test = np_utils.to_categorical(y_test, num_classes)

test_data = list(zip(X_test, y_test))

# initialize TensorFlow session
session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

# labels
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y, dimension=1)

# network graph params
filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 64

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 256


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(ipt, num_input_channels, conv_filter_size, num_filters):
    # define the weights
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    # create biases
    biases = create_biases(num_filters)
    # create a convolutional layer
    layer = tf.nn.conv2d(input=ipt, filter=weights, strides=[1, 1, 1, 1], padding='VALID')

    layer += biases

    # max-pooling.
    layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # output of pooling to relu activation function
    layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):
    # get shape of previous layer
    layer_shape = layer.get_shape()

    # number of features = img_height * img_width * num_channels
    num_features = layer_shape[1:4].num_elements()

    ## reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(ipt, num_inputs, num_outputs, use_relu=True):
    # define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # fully connected layer takes input x and produces wx+b
    # use matmul function since we're dealing with matrices
    layer = tf.matmul(ipt, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


layer_conv1 = create_convolutional_layer(ipt=x, num_input_channels=num_channels, conv_filter_size=filter_size_conv1, num_filters=num_filters_conv1)

layer_conv2 = create_convolutional_layer(ipt=layer_conv1, num_input_channels=num_filters_conv1, conv_filter_size=filter_size_conv2, num_filters=num_filters_conv2)

layer_conv3= create_convolutional_layer(ipt=layer_conv2, num_input_channels=num_filters_conv2, conv_filter_size=filter_size_conv3, num_filters=num_filters_conv3)

layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(ipt=layer_flat, num_inputs=layer_flat.get_shape()[1:4].num_elements(), num_outputs=fc_layer_size, use_relu=True)

layer_fc2 = create_fc_layer(ipt=layer_fc1, num_inputs=fc_layer_size, num_outputs=num_classes, use_relu=False)

y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y)

cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())

def show_progress(epoch, feed_dict_train, feed_dict_val, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_val)
    msg = 'Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}'
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


saver = tf.train.Saver()


def prepare_batch(data, batch_size):
    batch = sample(data, batch_size)
    images = [img for img, label in batch]
    X_batch = np.stack(images)
    y_batch = np.array([label for img, label in batch], dtype=np.int32)

    return X_batch, y_batch


def train(train_data, test_data):
    print('Training model...')
    n_epochs = 50
    batch_size = 20
    n_iters_per_epoch = len(train_data) // batch_size

    for epoch in range(n_epochs):
        for iteration in range(n_iters_per_epoch):
            x_batch, y_batch = prepare_batch(train_data, batch_size)
            feed_dict_tr = {x: x_batch, y: y_batch}
            session.run(optimizer, feed_dict=feed_dict_tr)

        x_valid_batch, y_valid_batch = prepare_batch(test_data, batch_size)
        feed_dict_val = {x: x_valid_batch, y: y_valid_batch}
        val_loss = session.run(cost, feed_dict=feed_dict_val)

        show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
        saver.save(session, 'trained_models/standard/tf_manual_model')

    # X_test, y_test = prepare_batch(test_data, batch_size=len(test_data))
    n_test_batches = 10
    X_test_batches = np.array_split(X_test, n_test_batches)
    y_test_batches = np.array_split(y_test, n_test_batches)

    acc_test = np.mean([session.run(accuracy, feed_dict={x: X_test_batch, y: y_test_batch}) for X_test_batch, y_test_batch in zip(X_test_batches, y_test_batches)])

    print('Test accuracy:', acc_test)


train(train_data, test_data)
