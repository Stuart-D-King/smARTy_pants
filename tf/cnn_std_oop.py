import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import np_utils
import cv2
from sys import argv
from scipy.misc import imresize
from random import sample
from numpy.random import seed
from tensorflow import set_random_seed
from sklearn.preprocessing import LabelEncoder
import pickle


def prepare_data():
    # prepare input data
    with open('data/class_dict.pkl', 'rb') as f:
        class_dict = pickle.load(f)

    classes = class_dict.values()
    num_classes = len(classes)

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

    return train_data, test_data


class ConvNN(object):

    def __init__(self, num_classes=10, img_size=299, num_channels=3, filter_size=3, num_filters=64, fc_layer_size=256):
        self.num_classes = num_classes
        self.img_size = img_size
        self.num_channels = num_channels
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.fc_layer_size = fc_layer_size

        # initialize TensorFlow session
        self.session = tf.Session()
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.num_channels], name='x')

        # labels
        self.y = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y')
        self.y_cls = tf.argmax(self.y, dimension=1)


    def build(self):
        layer_conv1 = self.create_convolutional_layer(ipt=self.x, num_input_channels=self.num_channels, conv_filter_size=self.filter_size, num_filters=self.num_filters)

        layer_conv2 = self.create_convolutional_layer(ipt=layer_conv1, num_input_channels=self.num_filters, conv_filter_size=self.filter_size, num_filters=self.num_filters)

        layer_conv3 = self.create_convolutional_layer(ipt=layer_conv2, num_input_channels=self.num_filters, conv_filter_size=self.filter_size, num_filters=self.num_filters)

        layer_flat = self.create_flatten_layer(layer_conv3)

        layer_fc1 = self.create_fc_layer(ipt=layer_flat, num_inputs=layer_flat.get_shape()[1:4].num_elements(), num_outputs=self.fc_layer_size, use_relu=True)

        layer_fc2 = self.create_fc_layer(ipt=layer_fc1, num_inputs=self.fc_layer_size, num_outputs=self.num_classes, use_relu=False)

        self.y_pred = tf.nn.softmax(layer_fc2, name='y_pred')

        self.y_pred_cls = tf.argmax(self.y_pred, dimension=1)

        self.session.run(tf.global_variables_initializer())
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=self.y)

        self.cost = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.cost)
        self.correct_prediction = tf.equal(self.y_pred_cls, self.y_cls)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()


    def create_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


    def create_biases(self, size):
        return tf.Variable(tf.constant(0.05, shape=[size]))


    def create_convolutional_layer(self, ipt, num_input_channels, conv_filter_size, num_filters):
        # define the weights
        weights = self.create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
        # create biases
        biases = self.create_biases(num_filters)
        # create a convolutional layer
        layer = tf.nn.conv2d(input=ipt, filter=weights, strides=[1, 1, 1, 1], padding='VALID')

        layer += biases

        # max-pooling.
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # output of pooling to relu activation function
        layer = tf.nn.relu(layer)

        return layer


    def create_flatten_layer(self, layer):
        # get shape of previous layer
        layer_shape = layer.get_shape()

        # number of features = img_height * img_width * num_channels
        num_features = layer_shape[1:4].num_elements()

        ## reshape to num_features
        layer = tf.reshape(layer, [-1, num_features])

        return layer


    def create_fc_layer(self, ipt, num_inputs, num_outputs, use_relu=True):
        # define trainable weights and biases.
        weights = self.create_weights(shape=[num_inputs, num_outputs])
        biases = self.create_biases(num_outputs)

        # fully connected layer takes input x and produces wx+b
        # use matmul function since we're dealing with matrices
        layer = tf.matmul(ipt, weights) + biases
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer


    def prepare_batch(self, data, batch_size):
        batch = sample(data, batch_size)
        images = [img for img, label in batch]
        X_batch = np.stack(images)
        y_batch = np.array([label for img, label in batch], dtype=np.int32)

        return X_batch, y_batch


    def show_progress(self, epoch, feed_dict_train, feed_dict_val, val_loss):
        acc = self.session.run(self.accuracy, feed_dict=feed_dict_train)
        val_acc = self.session.run(self.accuracy, feed_dict=feed_dict_val)
        msg = 'Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}'
        print(msg.format(epoch + 1, acc, val_acc, val_loss))


    def train(self, train_data, val_data, batch_size, n_epochs, save_path):
        print('Training model...')
        n_iters_per_epoch = len(train_data) // batch_size

        for epoch in range(n_epochs):
            for iteration in range(n_iters_per_epoch):
                x_batch, y_batch = self.prepare_batch(train_data, batch_size)
                feed_dict_tr = {self.x: x_batch, self.y: y_batch}
                self.session.run(self.optimizer, feed_dict=feed_dict_tr)

            x_valid_batch, y_valid_batch = self.prepare_batch(val_data, batch_size)
            feed_dict_val = {self.x: x_valid_batch, self.y: y_valid_batch}
            val_loss = self.session.run(self.cost, feed_dict=feed_dict_val)
            self.show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)

            self.saver.save(self.session, save_path)


    def predict(self, img_path, restore=False):
        with open('data/class_dict.pkl', 'rb') as f:
            class_dict = pickle.load(f)

        def _prepare_image(image, max_zoom=0.2):
            height = image.shape[0]
            width = image.shape[1]
            image_ratio = width / height
            target_image_ratio = self.img_size / self.img_size
            crop_vertically = image_ratio < target_image_ratio
            crop_width = width if crop_vertically else int(height * target_image_ratio)
            crop_height = int(width / target_image_ratio) if crop_vertically else height

            resize_factor = np.random.rand() * max_zoom + 1.0
            crop_width = int(crop_width / resize_factor)
            crop_height = int(crop_height / resize_factor)

            x0 = np.random.randint(0, width - crop_width)
            y0 = np.random.randint(0, height - crop_height)
            x1 = x0 + crop_width
            y1 = y0 + crop_height

            image = image[y0:y1, x0:x1]
            image = imresize(image, (self.img_size, self.img_size))

            return image.astype(np.float32) / 255

        image = cv2.imread(img_path, 1)
        image = _prepare_image(image)
        images = [image]
        # images.append(image)
        images = np.array(images)
        X_batch = images.reshape(1, self.image_size, self.image_size, self.num_channels)

        y_test_images = np.zeros((1, self.num_classes))

        if restore == True:
            # restore the saved model
            sess = tf.Session()
            # recreate the network graph
            saver = tf.train.import_meta_graph('trained_models/standard_oop/tf_model.meta')
            # load the weights saved using the restore graph
            saver.restore(sess, tf.train.latest_checkpoint('trained_models/standard_oop'))

            graph = tf.get_default_graph()

            y_pred = graph.get_tensor_by_name('y_pred:0')

            x = graph.get_tensor_by_name('x:0')
            y = graph.get_tensor_by_name('y:0')

            feed_dict_testing = {x: X_batch, y: y_test_images}
            result = sess.run(y_pred, feed_dict=feed_dict_testing)
        else:
            feed_dict_testing = {self.x: X_batch, self.y: y_test_images}
            result = self.sess.run(self.y_pred, feed_dict=feed_dict_testing)

        top_5 = np.argpartition(result[0], -5)[-5:]
        top_5 = reversed(top_5[np.argsort(result[0][top_5])])
        for i in top_5:
            print('{0}: {1:.2f}%'.format(class_dict[i], 100 * result[0][i]))


if __name__ == '__main__':
    seed(1337)
    set_random_seed(42)

    train_data, test_data = prepare_data()
    save_path = 'trained_models/standard_oop/tf_model'

    cnn = ConvNN(num_classes=10)
    cnn.build()
    cnn.train(train_data, test_data, batch_size=25, n_epochs=20, save_path=save_path)
    cnn.predict('images/9442.jpg', restore=False)
