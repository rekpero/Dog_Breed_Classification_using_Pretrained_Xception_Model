import urllib
import cv2
import pickle
import json
import os
import numpy as np
import matplotlib
import string
from sklearn.preprocessing import LabelEncoder
matplotlib.use('Agg')

# keras imports
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input
from keras.preprocessing import image
import matplotlib.pyplot as plt

# load the user configs
with open('conf/conf.json') as f:    
  config = json.load(f)

# config variables
model_name    = config["model"]
weights     = config["weights"]
include_top   = config["include_top"]
test_size     = config["test_size"]
model_path    = config["model_path"]
classifier_path = config["classifier_path"]

# create the pretrained models
# check for pretrained weight usage or not
# check for top layers to be included or not
if model_name == "vgg16":
  base_model = VGG16(weights=weights)
  model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
  image_size = (224, 224)
elif model_name == "vgg19":
  base_model = VGG19(weights=weights)
  model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
  image_size = (224, 224)
elif model_name == "resnet50":
  base_model = ResNet50(weights=weights)
  model = Model(input=base_model.input, output=base_model.get_layer('flatten').output)
  image_size = (224, 224)
elif model_name == "inceptionv3":
  base_model = InceptionV3(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
  model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
  image_size = (299, 299)
elif model_name == "inceptionresnetv2":
  base_model = InceptionResNetV2(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
  model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
  image_size = (299, 299)
elif model_name == "mobilenet":
  base_model = MobileNet(include_top=include_top, weights=weights, input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3))
  model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
  image_size = (224, 224)
elif model_name == "xception":
  base_model = Xception(weights=weights)
  model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
  image_size = (299, 299)
else:
  base_model = None

#loading pretrained weights
extractor_model = model
extractor_weights = model_path + str(test_size) + ".h5"
extractor_model.load_weights(extractor_weights)

#loading classifier model
with open(classifier_path, "rb") as classifier_file:
    classifier_model = pickle.load(classifier_file)

#user input URL
url = raw_input("Provide an url of Dog for Prediction...")
if not os.path.isdir("test_data") :
    os.system("mkdir " + "test_data")
urllib.urlretrieve(url, "test_data/test_dog.jpg")

#Path where test data is stored
image_path = "test_data/test_dog.jpg"
features = []

#Feature extraction of test data
img = image.load_img(image_path, target_size=image_size)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
feature = extractor_model.predict(x)
flat = feature.flatten()
features.append(flat)

#Classifier model
preds = classifier_model.predict(np.array(features))

#Decoding prediction
train_labels = os.listdir("train")
le = LabelEncoder()
le_labels = le.fit_transform(train_labels)


prediction = le.inverse_transform(preds)
pred = ''.join(prediction)
pred_clear = string.replace(pred, '_', ' ')
print("[INFO] Prediction for this given Photo is " + str(pred_clear))

#Showing the picture
plt.imshow(img)
plt.text(45, 180, '%s Prediction: %s' % (model_name, str(pred_clear)), color='w', backgroundcolor='k', alpha=1)
plt.savefig("predicted_fig.jpg")
plt.show()