'''
Sections of code adapted from Aurélien Géron's Hands-On Maching Learning with Scikit-Learn & TensorFlow textbook and accompanying Jupyter notebook for Chapter 13: Convolutional Neural Networks.
'''

import numpy as np
import pandas as pd
import os, cv2, pickle
from sys import argv
from random import sample
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim
from scipy.misc import imresize

'''
Prep, train, and make art image classification predictions using a TensorFlow CNN with an InceptionV3 baseline model.

The code can be run from the command line for one of three processes:
1. python tf_cnn_inception.py initial
        To create the necessary datasets and data files

2. python tf_cnn_inception.py train
        To train and save the CNN model

3. python tf_cnn_inception.py predict
        To predict the classification of a new art image - user is prompted to enter image file path after executing the above command
'''

INCEPTION_PATH = os.path.join("datasets", "inception")
INCEPTION_V3_CHECKPOINT_PATH = os.path.join(INCEPTION_PATH, "inception_v3.ckpt")

def reset_graph(seed=1337):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def img_path_class():
    img_root = 'img/'
    img_details = pd.read_csv('data/all_data_info.csv')
    keepers = ['Impressionism',
                'Expressionism',
                'Surrealism',
                'Cubism',
                'Abstract Art',
                'Fauvism',
                'Pop Art',
                'Art Deco',
                'Op Art',
                'Art Nouveau (Modern)']

    df_details = img_details[img_details['style'].isin(keepers)]
    img_names = df_details['new_filename'].values

    files = [f for f in os.listdir(img_root) if os.path.isfile(os.path.join(img_root, f))]

    art_list = []
    for name in files:
        if name in img_names:
            img_path = '{}{}'.format(img_root, name)
            art_list.append(img_path)

    names = []
    for path in art_list:
        img = cv2.imread(path, 1)
        try:
            img.shape
            names.append(path.lstrip(img_root))
        except AttributeError:
            continue

    styles = [df_details.loc[df_details['new_filename'] == name, 'style'].iloc[0] for name in names]

    images = ['{}{}'.format(img_root, name) for name in names]

    final_df = pd.DataFrame({'img_path':images, 'class':styles})
    final_df.to_pickle('paths_classes_10.pkl')


def sampled_paths_classes(df, size=200):
    # encode art categories as numerical values
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['class'].astype('str'))
    n_classes = len(np.unique(y))
    paths_and_classes = list(zip(df['img_path'].tolist(), y))

    paths_and_classes_small = []
    for x in range(n_classes):
        temp = [(path, style) for path, style in paths_and_classes if style == x]
        samp = sample(temp, size)
        for path, style in samp:
            paths_and_classes_small.append((path,style))

    np.random.shuffle(paths_and_classes_small)

    return paths_and_classes_small, encoder.classes_


def data_splits(paths_and_classes):
    test_ratio = 0.2
    train_size = int(len(paths_and_classes) * (1-test_ratio))

    train = paths_and_classes[:train_size]
    test = paths_and_classes[train_size:]

    return train, test


def cnn_with_inception(train_data, test_data):
    reset_graph()

    height = 299
    width = 299
    channels = 3

    X = tf.placeholder(tf.float32, shape=[None, height, width, channels], name='X')
    training = tf.placeholder_with_default(False, shape=[])
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(X, num_classes=1001, is_training=training)

    inception_saver = tf.train.Saver()
    prelogits = tf.squeeze(end_points['PreLogits'], axis=[1,2])
    n_outputs = 10

    with tf.name_scope('new_output_layer'):
        art_logits = tf.layers.dense(prelogits, n_outputs, name='art_logits')
        y_proba = tf.nn.softmax(art_logits, name='y_proba')

    y = tf.placeholder(tf.int32, shape=[None], name='y')

    with tf.name_scope('train'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=art_logits, labels=y)
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer()
        art_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='art_logits')
        training_op = optimizer.minimize(loss, var_list=art_vars)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(art_logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.name_scope('init_and_save'):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    n_epochs = 40
    batch_size = 25
    n_iters_per_epoch = len(train_data) // batch_size

    with tf.Session() as sess:
        init.run()
        inception_saver.restore(sess, INCEPTION_V3_CHECKPOINT_PATH)

        for epoch in range(n_epochs):
            print('Epoch', epoch, end='')
            for iteration in range(n_iters_per_epoch):
                print('.', end='')
                X_batch, y_batch = prepare_batch(train_data, batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})

            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            print(' Training accuracy:', acc_train)

            save_path = saver.save(sess, 'trained_models/inception/tf_inception')

        X_test, y_test = prepare_batch(test_data, batch_size=len(test_data))

        n_test_batches = 10
        X_test_batches = np.array_split(X_test, n_test_batches)
        y_test_batches = np.array_split(y_test, n_test_batches)

        with tf.Session() as sess:
            saver.restore(sess, 'trained_models/inception/tf_inception')

            print('Computing final accuracy on the test set...')
            acc_test = np.mean([accuracy.eval(feed_dict={X: X_test_batch, y: y_test_batch}) for X_test_batch, y_test_batch in zip(X_test_batches, y_test_batches)])
            print('Test accuracy:', acc_test)


def make_prediction(img_path):
    with open('data/class_dict.pkl', 'rb') as f:
        class_dict = pickle.load(f)

    image = cv2.imread(img_path, 1)
    image = prepare_image(image, prediction=True)
    images = []
    images.append(image)
    images = np.array(images)
    image_size = 299
    n_channels = 3
    X_batch = images.reshape(1, image_size, image_size, n_channels)

    # restore the saved model
    sess = tf.Session()
    # recreate the network graph
    saver = tf.train.import_meta_graph('trained_models/inception/tf_inception.meta')
    # load the weights saved using the restore graph
    saver.restore(sess, tf.train.latest_checkpoint('trained_models/inception/'))

    graph = tf.get_default_graph()

    y_pred = graph.get_tensor_by_name('new_output_layer/y_proba:0')

    x = graph.get_tensor_by_name('X:0')
    y = graph.get_tensor_by_name('y:0')
    y_test_images = np.zeros((1,))

    feed_dict_testing = {x: X_batch, y: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)

    top_5 = np.argpartition(result[0], -5)[-5:]
    top_5 = reversed(top_5[np.argsort(result[0][top_5])])
    print('Top 5 Predictions:')
    print('------------------')
    for i in top_5:
        print('{0}: {1:.2f}%'.format(class_dict[i], 100 * result[0][i]))


def prepare_batch(data, batch_size):
    batch_paths_and_classes = sample(data, batch_size)

    images = [img for img, label in batch_paths_and_classes]
    X_batch = 2 * np.stack(images) - 1 # needed for inception V3
    y_batch = np.array([label for img, label in batch_paths_and_classes], dtype=np.int32)

    return X_batch, y_batch


def prepare_image(image, target_width=299, target_height=299, max_zoom=0.2, prediction=False):
    height = image.shape[0]
    width = image.shape[1]
    image_ratio = width / height
    target_image_ratio = target_width / target_height
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

    if not prediction:
        if np.random.rand() < 0.5:
            image = np.fliplr(image)

    image = imresize(image, (target_width, target_height))

    return image.astype(np.float32) / 255


def initial_run():
    img_path_class()
    df = pd.read_pickle('data/paths_classes_10.pkl')

    paths_and_classes_small, class_names = sampled_paths_classes(df)

    with open('data/paths_and_classes_small.pkl', 'wb') as f:
        pickle.dump(paths_and_classes_small, f)

    class_dict = {index: art_class for index, art_class in zip(range(10), class_names)}

    with open('data/class_dict.pkl', 'wb') as f:
        pickle.dump(class_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    images = [cv2.imread(path, 1) for path, label in paths_and_classes_small]

    x = np.array([prepare_image(image) for image in images])
    y = np.array([style for path, style in paths_and_classes_small])

    np.savez('data/images_labels_299.npz', x=x, y=y)
    x = data['x']
    y = data['y']
    data = list(zip(x,y))

    train_data, test_data = data_splits(data)

    with open('data/train_data_299.pkl', 'wb') as f:
        pickle.dump(train_data, f)

    with open('data/test_data_299.pkl', 'wb') as f:
        pickle.dump(test_data, f)


if __name__ == '__main__':
    _, run_type = argv

    if run_type.lower() not in ['inital', 'train', 'predict']:
        print('Please run the code in one of three ways: \n- python tf_cnn_inception.py initial \n- python tf_cnn_inception.py train \n- python tf_cnn_inception.py predict')

    if run_type.lower() == 'initial':
        initial_run()

    if run_type.lower() == 'train':
        with open('data/train_data_299.pkl', 'rb') as f:
            train_data = pickle.load(f)

        with open('data/test_data_299.pkl', 'rb') as f:
            test_data = pickle.load(f)

        cnn_with_inception(train_data, test_data)

    if run_type.lower() == 'predict':
        img_path = input('Please type image file path: ')
        make_prediction(img_path)
