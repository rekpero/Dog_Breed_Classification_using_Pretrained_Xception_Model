# Dog_Breed_Classification_using_Pretrained_Xception_Model

This code is for "Dog Breed Classification using Pretrained Models". In this code, we will be using the pre-trained Deep Neural Nets, which is trained on the ImageNet challenge that are made publicly available in Keras. We will specially use Kaggle Dataset of Flower Recognition.
The pre-trained models we will consider are VGG16, VGG19, Inception-v3, Xception, ResNet50, InceptionResNetv2 and MobileNet. We are mainly using "Pretrained Xception Model".
# Organizing the Dataset
The dataset you will be provided is in Raw Form. So its better to Organize the dataset first. For this you have to run the python program 'data_organizer.py'. This will automatically copy the image to the destination path. Also you have to input relevent info regarding 'path to raw dataset', 'path to training dataset'. The CSV file named 'labels.csv' has all image ids and its respective labels so we will take image from source path to copy it to newly created label directory.
# Model Selection
Since, we have many pre-trained models so you have to choose which Model you want to use to Extract features. For this, you have to run the python program 'model_selector.py'. This code will give you options and you have to input the model name for extracting features from training images. Also you have to input relevent info regarding 'number of classes', 'test size' and 'path to training dataset'. This code basically update the 'conf.json' file as per user input. 
### WARNING: YOU HAVE TO CLONE THE 'CONF/CONF.JSON' FILE OTHERWISE MODEL SELECTOR WON'T WORK.
# Feature Extraction using ConvNets
Traditional machine learning approach uses feature extraction for images using Global descriptors such as Local Binary Patterns, Histogram of Oriented Gradients, Color Histograms etc. or Local descriptors such as SIFT, SURF, ORB etc. Instead of using hand-crafted features, Deep Neural Nets automatically learns these features from images in a hierarchical fashion. 
Thus, we can use a Convolutional Neural Network as a Feature Extractor by taking the activations available before the last fully connected layer of the network (i.e. before the final softmax classifier). These activations will be acting as the feature vector for a machine learning classifier which further learns to classify it. This type of approach is well suited for Image Classification problems, where instead of training a CNN from scratch (which is time-consuming and tedious), a pre-trained CNN could be used as a Feature Extractor.
For extracting features using ConvNets, you have to run the python program 'features_extractor.py'. This will automatically encode the Labels and store labels and features in the disk in '.h5' format. The weigths are also saved in the disks in both '.json' and '.h5' format. The code will automatically generate 'output' folder and save labels, features and model in respective models folder.
# Classifying the Dataset
Basically, Logistic Regression is used as classifier in this code. The tensor for features is 1D so Logistic Regression will fit best in this case. For classifying, you have to run the python program 'dog_breed_classifier.py'. This will automatically train the classifier and find the accuracy from 1st Rank and 5th Rank, also a classificaton report and save it to 'Result.txt' file. It also generate a confusion matrix with is saved as 'confusion_matrix.jpg'.
# Predictng Flowers directly from URL
In this code, I have added a user input url for prediction. For this, you have to run the python program 'prediction.py' where you have to provide a url link having image of any flower. This will automatically retrive the image and save it to local memory. Then, it will automatically load the image and extract its features same way as earlier. And it will predict the class for the image. The picture will be displayed or saved in the local machine with its predicted value.

# Dependencies
You will need the following Python packages to run the code.
1.  Theano or TensorFlow
2.  Keras
3.  NumPy
4.  Scikit-Learn
5.  Matplotlib
6.  Seaborn
7.  h5py
8.  Dataset used here is https://www.kaggle.com/c/dog-breed-identification
