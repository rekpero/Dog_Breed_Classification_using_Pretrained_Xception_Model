# organize imports
import os
import datetime
import shutil
import pandas as pd
import numpy as np

# print start time
print ("[INFO] program started on - " + str(datetime.datetime.now))

# get the input and output path
input_img_path = raw_input("Enter the path of raw dataset you want to transfer...\n")
input_path  =  "../" + input_img_path
output_img_path = raw_input("Enter the path to the training dataset...\n")
#creating train directory
if not os.path.isdir(output_img_path) :
        os.system("mkdir " + output_img_path)
output_path = output_img_path

labels = pd.read_csv("labels.csv")
#labels.head()
labels = np.array(labels)
print(labels.shape)

# change the current working directory
os.chdir(output_path)

#number of labels
num_labels = 0

for x in range(0, len(labels)):
    # create a folder for that class
    if not (os.path.isdir(labels[x][1])):
        os.system("mkdir " + labels[x][1])
        num_labels = num_labels + 1
    # get the current path
    cur_path = labels[x][1] 
    # get image path
    image_path = input_path 
    # loop over the images in the dataset
    image_name = os.path.join(image_path, labels[x][0] + ".jpg")
    shutil.copy(image_name, cur_path)
    
print("Number of Labels is " + str(num_labels))
    
# print end time
print ("[INFO] program ended on - " + str(datetime.datetime.now))