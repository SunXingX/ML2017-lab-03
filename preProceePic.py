import os
import pickle

import numpy as np
from PIL import Image

from feature import NPDFeature


def convertToGray(path, savePath):
    for file in os.listdir(path):
        #print(file)
        file_path = os.path.join(path, file)
        #print("文件路径是：", file_path)
        #file is dir
        if os.path.isdir(file_path):
            convertToGray(file_path, savePath)
        #file is regular file
        if os.path.isfile(file_path):
            #print("is file", file)
            if os.path.splitext(file)[1] == '.jpg':
                #print("is jpg", file)
                im = Image.open(file_path)
                im_gray = im.convert("L")

                parent_path_name = os.path.dirname(file_path).split("/")[-1]
                save_path = os.path.join(savePath, parent_path_name)
                save_path = os.path.join(save_path, file)
                #print(save_path)
                length = 24
                width = 24
                if not os.path.exists(save_path):
                    im_gray = im_gray.resize((length, width))
                    im_gray.save(save_path)
    return True


def getFeature(Path, savePath):
    for file in os.listdir(Path):
        file_abs_path = os.path.join(Path, file)
        if os.path.isdir(file_abs_path):
            getFeature(file_abs_path, savePath)
        if os.path.isfile(file_abs_path):
            if os.path.exists(file_abs_path):
                #using NPDFeature class to get feature
                im = np.array(Image.open(file_abs_path))
                #print("image.shape = ", im.shape)
                #print("**8*************======", im)
                npdf = NPDFeature(im)
                pic_features = npdf.extract()
                print("pic_features", pic_features.shape)

                #to get path for saving features as files
                parent_path_name = os.path.dirname(file_abs_path).split("/")[-1]
                save_path = os.path.join(savePath, parent_path_name)
                save_path = os.path.join(save_path, os.path.splitext(file)[0])

                if not os.path.exists(save_path):
                    #dump to file 
                    output = open(save_path, "wb")
                    PROROCOL = 0
                    pickle.dump(pic_features, output, PROROCOL)


def mergeFeature(Path, X_save_name, y_save_name):
    is_first = True
    dataset_X = 0
    dataset_y = np.ones((1, 1))

    for root, dirs, files in os.walk(Path):
        for name in files:
            print(os.path.join(root, name))
            file_abs_path = os.path.join(root, name)

            #load features from files using dump
            pkl_file = open(file_abs_path, 'rb')
            pic_features = pickle.load(pkl_file)
            m = pic_features.shape[0]
            pic_features = pic_features.reshape((m, 1)).T

            type = name.strip().split("_")[0]

            if (is_first):
                dataset_X = pic_features
                if (type == "face"):
                    dataset_y = np.ones((1, 1))
                elif(type == "nonface"):
                    dataset_y = np.ones((1, 1)) - 2
                is_first = False
            else:
                dataset_X = np.row_stack((dataset_X, pic_features))
                
                #print("&&&&&&&&&&&&&&&&&&&&&&", name.strip().split("_")[0])
                if (type == "face"):
                    dataset_y = np.row_stack((dataset_y, np.ones((1, 1))))
                elif(type == "nonface"):
                    dataset_y = np.row_stack((dataset_y, np.ones((1, 1)) - 2))
            print("dataset_X", dataset_X.shape)
            #print("dataset_y", dataset_y.shape)
            print("dataset_y", dataset_y.shape)

    save_dataset_x = os.path.join(Path, X_save_name)
    save_dataset_y = os.path.join(Path, y_save_name)
    
    if not os.path.exists(save_dataset_x):
        pkl_save_train_x = open(save_dataset_x,  "wb")
        pickle.dump(dataset_X, pkl_save_train_x)
    if not os.path.exists(save_dataset_y):
        pkl_save_train_y = open(save_dataset_y, "wb")
        pickle.dump(dataset_y, pkl_save_train_y)    




save_path = "/home/sun/ComputerScience/MachineLearning/Experiments/Experiment_three/ML2017-lab-03/datasets/gray/"
pic_path = "/home/sun/ComputerScience/MachineLearning/Experiments/Experiment_three/ML2017-lab-03/datasets/original/"
#convertToGray(pic_path, save_path)

trainset_save_path = "/home/sun/ComputerScience/MachineLearning/Experiments/Experiment_three/ML2017-lab-03/datasets/features/"
#getFeature(save_path, feature_save_path)

#mergeFeature(trainset_save_path, "X_train", "y_train")

validset_save_path = "/home/sun/ComputerScience/MachineLearning/Experiments/Experiment_three/ML2017-lab-03/datasets/valid/"
mergeFeature(validset_save_path, "x_valid", "y_valid")

