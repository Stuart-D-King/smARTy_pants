import numpy as np
import pandas as pd
from random import sample
import pickle, cv2
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
from keras import applications, optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras.utils import np_utils


def train_validation_split(x, y):
    # split data into training and test sets
    X_training, X_test, y_training, y_test = train_test_split(x, y, stratify=y, random_state=1337)

    # split training into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, stratify=y_training, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


def one_hot(y_train, y_val, y_test, n_classes):
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_val = np_utils.to_categorical(y_val, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)

    return y_train, y_val, y_test


def build_fit_save_cnn(input_shape, n_classes, epochs, batch_size, X_train, X_val, y_train, y_val):
    base_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    add_model.add(Dense(512, activation='relu'))
    add_model.add(Dropout(0.25))
    add_model.add(Dense(n_classes, activation='softmax'))

    # combine base model and fully connected layers
    final_model = Model(inputs=base_model.input, outputs=add_model(base_model.output))

    # specify SDG optimizer parameters
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    # compile model
    final_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    final_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_val, y_val))

    score = final_model.evaluate(X_val, y_val, verbose=0)
    print('Val. score:', score[0])
    print('Val. accuracy:', score[1])

    save_model(final_model)

    return final_model


def test_predict_score(model, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose=0)
    test_pred = model.predict(X_test)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    return test_pred, score


def save_model(model):
    model_json = model.to_json()
    with open('saved_model/resnet_model.json', 'w') as json_file:
        json_file.write(model_json)

    model.save_weights('saved_model/resnet_model_weights.h5')
    print('Model saved to disk!')


if __name__ == '__main__':
    seed = 1337
    np.random.seed(seed)

    epochs = 30
    batch_size = 25
    input_shape = (224,224,3)

    data = np.load('data/images_labels_224.npz')
    x = data['x']
    y = data['y']
    n_classes = len(np.unique(y))

    # train/validation split
    X_train, X_val, X_test, y_train, y_val, y_test = train_validation_split(x, y)

    # convert y to one-hot encoding
    y_train, y_val, y_test = one_hot(y_train, y_val, y_test, n_classes)

    # build, train, and save CNN model
    final_model = build_fit_save_cnn(input_shape, n_classes, epochs, batch_size, X_train, X_val, y_train, y_val)

    # score model on test set
    test_pred, score = test_predict_score(loaded_model, X_test, y_test)
