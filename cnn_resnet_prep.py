'''
Sections of code adapted from Aurélien Géron's Hands-On Maching Learning with Scikit-Learn & TensorFlow textbook and accompanying Jupyter notebook for Chapter 13: Convolutional Neural Networks.
'''

import numpy as np
import pandas as pd
from random import sample
from scipy.misc import imresize
import pickle, cv2
from sklearn.preprocessing import LabelEncoder

'''
Read in images and accompanying metadata to create sampled dataset consisting of 200 images from the ten art styles of interest. Save processed datasets to file.
'''

def make_img_df():
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
    final_df.to_pickle('data/paths_classes_10.pkl')


def prepare_data():
    df = pd.read_pickle('data/paths_classes_10.pkl')

    paths_and_classes_small, class_names = sampled_paths_classes(df)

    with open('data/paths_and_classes_small.pkl', 'wb') as f:
        pickle.dump(paths_and_classes_small, f)

    class_dict = {index: art_class for index, art_class in zip(range(10), class_names)}

    with open('data/class_dict.pkl', 'wb') as f:
        pickle.dump(class_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    images = [cv2.imread(path,1) for path, label in paths_and_classes_small]

    x = np.array([prepare_image(image) for image in images])
    y = np.array([style for path, style in paths_and_classes_small])

    np.savez('data/images_labels_224.npz', x=x, y=y)


def sampled_paths_classes(df):
    # encode art categories as numerical values
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['class'].astype('str'))
    n_classes = len(np.unique(y))
    paths_and_classes = list(zip(df['img_path'].tolist(), y))

    paths_and_classes_small = []
    for x in range(n_classes):
        temp = [(path, style) for path, style in paths_and_classes if style == x]
        samp = sample(temp, 200)
        for path, style in samp:
            paths_and_classes_small.append((path,style))

    np.random.shuffle(paths_and_classes_small)

    return paths_and_classes_small, encoder.classes_


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

    if np.random.rand() < 0.5:
        image = np.fliplr(image)

    image = imresize(image, (target_width, target_height))

    return image.astype(np.float32) / 255


if __name__ == '__main__':
    make_img_df()
    prepare_data()
