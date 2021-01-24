# Facial-Keypoint-Detection

# Project Overview
In this project, you’ll combine your knowledge of computer vision techniques and deep learning architectures to build a facial keypoint detection system that takes in any 
image with faces, and predicts the location of 68 distinguishing keypoints on each face!

Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, 
facial pose recognition, facial filters, and emotion recognition. Your completed code should be able to look at any image, detect faces,
and predict the locations of facial keypoints on each face. Some examples of these keypoints are pictured below.

# Project Instructions
The project will be broken up into a few main parts in four Python notebooks, 
only Notebooks 2 and 3 (and the models.py file) will be graded:

Notebook 1 : Loading and Visualizing the Facial Keypoint Data

Notebook 2 : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

Notebook 3 : Facial Keypoint Detection Using Haar Cascades and your Trained CNN

Notebook 4 : Fun Filters and Keypoint Uses

You can find these notebooks in the Udacity workspace that appears in the concept titled Project: Facial Keypoint Detection. 
This workspace provides a Jupyter notebook server directly in your browser.

# Define the Network Architecture
Define a data_transform and apply it whenever you instantiate a DataLoader. The composed transform should include: rescaling/cropping, normalization, 
and turning input images into torch Tensors. The transform should turn any input image into a normalized, square, grayscale image and then a Tensor for your model 
to take it as input.

Depending on the complexity of the network you define, and other hyperparameters the model can take some time to train. We encourage you to start with a simple 
network with only 2 layers. You'll be graded based on the implementation of your models rather than accuracy.

# Facial Keypoint Detection
Use a Haar cascade face detector to detect faces in a given image.

You should transform any face into a normalized, square, grayscale image and then a Tensor for your model to 
take in as input (similar to what the data_transform did in Notebook 2).

After face detection with a Haar cascade and face pre-processing, apply your trained model to each detected face, 
and display the predicted keypoints for each face in the image.
