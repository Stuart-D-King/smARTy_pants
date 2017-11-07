'''
Code adapted from Ankit Sachan's Tensorflow Tutorial 2: image classifier using convolutional neural network blog post (http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/) and GitHub repository (https://github.com/sankit1/cv-tricks.com/tree/master/Tensorflow-tutorials/tutorial-2-image-classifier)
'''

import tensorflow as tf
import numpy as np
import cv2, pickle
from sys import argv
from scipy.misc import imresize

'''
To make a prediction from the command line, type:
    python cnn_std_predict.py path_to_image
'''

_, img_path = argv

image_size = 299
num_channels = 3
images = []

with open('data/class_dict.pkl', 'rb') as f:
    class_dict = pickle.load(f)


def prepare_image(image, target_width=299, target_height=299, max_zoom=0.2):
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
    image = imresize(image, (target_width, target_height))

    return image.astype(np.float32) / 255

image = cv2.imread(img_path, 1)
image = prepare_image(image)
images = []
images.append(image)
images = np.array(images)
X_batch = images.reshape(1, image_size, image_size, num_channels)

# restore the saved model
sess = tf.Session()
# recreate the network graph
saver = tf.train.import_meta_graph('trained_models/standard/tf_manual_model.meta')
# load the weights saved using the restore graph
saver.restore(sess, tf.train.latest_checkpoint('trained_models/standard/'))

graph = tf.get_default_graph()

y_pred = graph.get_tensor_by_name('y_pred:0')

x = graph.get_tensor_by_name('x:0')
y = graph.get_tensor_by_name('y:0')
y_test_images = np.zeros((1, 10))

feed_dict_testing = {x: X_batch, y: y_test_images}
result = sess.run(y_pred, feed_dict=feed_dict_testing)

top_5 = np.argpartition(result[0], -5)[-5:]
top_5 = reversed(top_5[np.argsort(result[0][top_5])])
for i in top_5:
    print('{0}: {1:.2f}%'.format(class_dict[i], 100 * result[0][i]))
