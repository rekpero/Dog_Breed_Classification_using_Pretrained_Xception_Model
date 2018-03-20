import os
import json

# load the user configs
with open('conf/conf.json', 'r+') as f:
    config = json.load(f)
    print("Which Model do you want to train...")
    model_input = raw_input("Options: vgg16 \nvgg19 \nresnet50 \ninceptionv3 \ninceptionresnetv2 \nmobilenet \nxception \n")
    print("[INFO] You have chosen: " + model_input)
    curr_dir = "output"
    if not os.path.isdir(curr_dir) :
        os.system("mkdir " + curr_dir)
    if not os.path.isdir(curr_dir + "/" + model_input) :
        os.system("mkdir " + curr_dir + "/" + model_input)
    config["model"] = model_input
    config["classifier_path"] = "output/" + model_input + "/classifier.pickle"
    config["features_path"] = "output/" + model_input + "/features.h5"
    config["labels_path"] = "output/" + model_input + "/labels.h5"
    config["model_path"] = "output/" + model_input + "/model"
    config["results"] = "output/" + model_input + "/results.txt"
    num_classes_input = raw_input("Enter the no. of classes to train...")
    config["num_classes"] = int(num_classes_input)
    test_size_input = raw_input("Enter the test size to train...")
    config["test_size"] = float(test_size_input)
    train_path_input = raw_input("Enter the path of training dataset...")
    config["train_path"] = train_path_input
    test_path_input = raw_input("Enter the path of test dataset...")
    config["test_path"] = test_path_input
    config["include_top"] = "false"
    config["seed"] = 9
    config["weigths"] = "imagenet"

    print("\n[INFO] Configuration you have chosen for training: \n")
    print (config)
    f.seek(0)
    json.dump(config, f, indent=4)
    f.truncate()