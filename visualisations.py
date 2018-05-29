from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from os.path import join, isfile, isdir
from os import listdir

# ---
# Problem 1
# ---------
#
# Let's take a peek at some of the data to make sure it looks sensible. Each exemplar should be an image of a character A through J rendered in a different font. Display a sample of the images that we just downloaded. Hint: you can use the package IPython.display.
#
# ---

#PROBLEM1 GOOD
def displayImagesFromTrainFolders(train_folders):
    MAX_SELECTION_FROM_CATEGORY = 1;
    MAX_CATEGORIES = 3;
    permut = np.random.permutation(len(train_folders));
    permut = permut[0:MAX_CATEGORIES];
    print(permut);

    for index in permut:
        # open the training data set coresponding letter and display a random picture
        onlyFiles =  listdir(train_folders[index]);
        onlyFiles = onlyFiles[0:MAX_SELECTION_FROM_CATEGORY];
        for file in onlyFiles:
            completeFileName = join(train_folders[index], file);
            print(completeFileName);
            display(Image(filename=completeFileName));


#PROBLEM 2 GOOD
def showImagesFromAllPickleFolders(train_datasets, test_datasets):
    MAX_SELECTION_FROM_CATEGORY = 1;
    for file in train_datasets + test_datasets:
        with open(file, 'rb') as f:
            dataset = pickle.load(f);
            permut = np.random.permutation(np.shape(dataset)[0]);
            permut = permut[:MAX_SELECTION_FROM_CATEGORY];

            label = str(os.path.splitext(os.path.splitext(file)[0])[0]);
            label = label[-1];

            fig = plt.figure()

            count = 0;
            for index in permut:
                count = count + 1;
                ax1 = fig.add_subplot(1, MAX_SELECTION_FROM_CATEGORY, count);
                ax1.set_title(label);
                ax1.imshow(dataset[index])

def getLabel(filename):
    label = str(os.path.splitext(os.path.splitext(filename)[0])[0]);
    label = label[-1];
    return ord(label)-ord("A");


#PROBLEM3 GOOD
def showHistogramsForTrainTest(train_datasets, test_datasets):
    nImagesDistribution = np.zeros(len(train_datasets));
    # list of pickle files.
    for file in train_datasets:
        count = 0;
        with open(file, 'rb') as f:
            dataset = pickle.load(f);
            nDataSet = np.shape(dataset)[0];
            nImagesDistribution[count] = nDataSet;
            count = count + 1;

    plt.hist(nImagesDistribution, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram of training data number of images")
    plt.show()

    for file in test_datasets:
        count = 0;
        with open(file, 'rb') as f:
            dataset = pickle.load(f);
            nDataSet = np.shape(dataset)[0];
            nImagesDistribution[count] = nDataSet;
            count = count + 1;

    plt.hist(nImagesDistribution, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram of test data number of images")
    plt.show()



# Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray. Hint: you can use matplotlib.pyplot.
#PROBLEM 4 GOOD
def showImageFromDataSet(imageset, labelset, maxSelections):
    permut = np.random.permutation(np.shape(imageset)[0]);
    permut = permut[:maxSelections];
    fig = plt.figure()
    count = 0;
    for index in permut:
        count = count + 1;
        ax1 = fig.add_subplot(1, maxSelections, count);
        ax1.set_title(chr((int(labelset[index]+97))));
        ax1.imshow(imageset[index])







