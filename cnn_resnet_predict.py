'''
Sections of code adapted from Aurélien Géron's Hands-On Maching Learning with Scikit-Learn & TensorFlow textbook and accompanying Jupyter notebook for Chapter 13: Convolutional Neural Networks.
'''

import numpy as np
from sys import argv
from scipy.misc import imresize
import pickle, cv2
from keras import optimizers
from keras.models import load_model
from keras.models import model_from_json

'''
Load a pre-trained Keras CNN model and make a art style prediction for a new image. Predictions can be made by typing the following in the command line:
    python make_prediction.py path_to_image
'''

def load_model():
    json_file = open('saved_model/resnet_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('saved_model/resnet_model_weights.h5')

    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return loaded_model


def make_prediction(img_path):
    model = load_model()

    with open('data/class_dict.pkl', 'rb') as f:
        class_dict = pickle.load(f)

    image = cv2.imread(img_path, 1)
    image = prepare_image(image)
    images = []
    images.append(image)
    images = np.array(images)
    image_size = 224
    n_channels = 3
    X_batch = images.reshape(1, image_size, image_size, n_channels)

    preds = model.predict(X_batch)

    top_5 = np.argpartition(preds[0], -5)[-5:]
    top_5 = reversed(top_5[np.argsort(preds[0][top_5])])
    print('Top 5 Predictions:')
    print('------------------')
    for i in top_5:
        print('{0}: {1:.2f}%'.format(class_dict[i], 100 * preds[0][i]))


def prepare_image(image, target_width=224, target_height=224, max_zoom=0.2):
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


if __name__ == '__main__':
    _, img_path = argv
    make_prediction(img_path)
