## smARTy pants
### Predicting modern art styles using image recognition  
Created by: **Stuart King**  
November 2017

### Overview
I've always wanted to able to speak intellectually artwork. Sadly, despite my college Art Appreciation professor's best efforts, the nuances of modern art styles never truly sunk in. In an attempt to use a machine to make me sound (artificially) smarter than I am about art, I have created an art image classifier to predict one of ten modern art styles a piece of art would fall into. Below is a description of the steps taken and results achieved.

### Data Wrangling
The Kaggle [Painter By Numbers](https://www.kaggle.com/c/painter-by-numbers) competition challenges participants to use computer vision to examine pairs of paintings and determine if they are by the same artist. Included in this challenge is a dataset of over 100,000 images with image-specific metadata, including the particular style of art the image is classified into. Using this large dataset, I sampled 200 images from each style I wanted to focus on - **Impressionism, Expressionism, Surrealism, Cubism, Abstract Art, Fauvism, Pop Art, Art Deco, Op Art, and Art Nouveau (Modern)** - to create my final project dataset of 2,000 images.

To prepare the data, I pre-processed each image to convert the image into a normalized, pixel array of shape 224 x 224 x 3. Image labels (art styles) were encoded into a range of values between 0 and 9 before being converted into one-hot vector arrays.

Finally, the prepared dataset of image pixel arrays and corresponding one-hot categorical vectors were split into training, validation, and test sets, consisting of 1,125, 375, and 500 observations, respectfully.

### Modeling
I used the **Keras** neural network API with a Tensorflow backend to construct my Convolutional Neural Network's (CNN) architecture. I decided to use Keras in part due to the deep learning models and their pre-trained weights that Keras has made available for public use. In particular, I wanted to use the ResNet50 model with weights pre-trained on ImageNet dataset. Because ResNet50 was trained on millions of images of objects, it is already trained to detect basic features such as edges and colors. Using this baseline model and its corresponding weights, I was then able to add fully connected layers specific to my image dataset to fine tune ResNet50 and apply its understanding of basic objects to identify features that distinguish different art styles. The general CNN architecture is as follows:

**Base Model:**  
ResNet50 trained on ImageNet dataset  

**Fully Connected Layers:**
- Flatten
- Dense (512 neurons, activation = relu)
- Dense (10 classes of art, activation = softmax)  

**Compiled Model:**
- Optimizer = stochastic gradient descent (SGD)
- Loss = categorical cross-entropy

### Results
After training the model for 30 epochs of batch size 25 using an AWS DeepLearning AMI CUDA 8 Ubuntu EC2 instance, my model produced an accuracy score on the test dataset of **57.8 percent**. This certainly was not as high as I had hoped, but it is important to temper expectations given the difficult of the computer-vision task. Random guessing would result in an average correct style classification of 10 percent, thus 58 percent accuracy might not be that bad...

To further example the model's output, I predicted the style classification of three images.

#### Image #1  
**Actual style:**  
Abstract Art  

**Top 5 Predictions:**  
Pop Art: 48.64%  
Art Nouveau (Modern): 18.67%  
Surrealism: 9.60%  
Abstract Art: 7.16%  
Art Deco: 4.39%

![abstract](images/35840.jpg)

#### Image #2
**Actual style:**  
Surrealism

**Top 5 Predictions:**  
Expressionism: 76.49%  
Surrealism: 11.88%  
Abstract Art: 4.36%  
Fauvism: 2.66%  
Cubism: 1.89%  

![surrealism](images/64423.jpg)

#### Image #3
**Actual style:**  
Pop Art

**Top 5 Predictions:**  
Pop Art: 96.31%  
Op Art: 2.33%  
Abstract Art: 1.01%  
Expressionism: 0.28%  
Cubism: 0.03%  

![pop](images/9442.jpg)

As is evident from the above, the model is far from perfect, but it does seem to perform better depending on the particular art style. The first prediction for the abstract art piece was far off, with three other styles being predicted above the true class. The model did a bit better for the surrealism art image, predicting surrealism as the second most likely category. Finally, we see a strong performance from the model in respects to its prediction for the tested pop art image, outputting a 96 percent probability that the image belongs to the pop art style.

### Next Steps
While the model demonstrates some promise, it has a long way to go before anyone would be able to confidently trust its output. Tweaks to the model's architecture make sense for immediate next steps, including the addition of supplemental fully connected layers, and automatic adjustments to the learning rate as model performance plateaus. I am also currently working on a pure **TensorFlow** implementation of the model using transfer learning from the Inception V3 pre-trained model. Once completed, I will be able to compare the two models to determine which is better and worthy of further improvements.

### Make Predictions
To make predictions on new images, you will need to download and save in a consolidated location the following in the same file folder format:

```
+-- cnn_resnet_predict.py
+-- data/
    +-- class_dict.pkl
+-- saved_model/
    +-- resnet_model_weights.h5
    +-- resnet_model.json
```
Once the necessary files and folders are compiled, you can make predictions using the command line as follows:
```
python cnn_resnet_predict.py path_to_image
```

### Acknowledgements
Code used in this project was adapted from Jen Waller's [Wildflower Finder](https://github.com/jw15/wildflower-finder) project, and Aurélien Géron's **Hands-On Maching Learning with Scikit-Learn & TensorFlow** and accompanying Jupyter notebook for Chapter 13: Convolutional Neural Networks.
