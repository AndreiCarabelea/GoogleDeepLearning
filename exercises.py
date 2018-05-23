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

# In[6]:
#Problem 1
# generate a permutation of letters.
def displayImagesFromTrainFolders(train_folders):
    MAX_SELECTION_FROM_CATEGORY = 1;
    MAX_CATEGORIES = 3;
    permut = np.random.permutation(10);
    permut = permut[0:MAX_CATEGORIES];
    print(permut);

    for index in permut:
        # open the training data set coresponding letter and display a random picture
        onlyfiles = [f for f in listdir(train_folders[index]) if isfile(join(train_folders[index], f))]
        onlyfiles = onlyfiles[0:MAX_SELECTION_FROM_CATEGORY];
        for file in onlyfiles:
            completeFileName = join(train_folders[index], file);
            print(completeFileName);
            # displays a np.array of 28x28 integers
            print(imageio.imread(completeFileName).astype(float).dtype)
            display(Image(filename=completeFileName));


# ---
# Problem 2
# ---------
#
# Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray. Hint: you can use matplotlib.pyplot.
#
# ---

# In[8]:

# Solution for problem 2

def showImageFromDataSet(imageset, labelset):
    MAX_SELECTION_FROM_CATEGORY = 1;
    permut = np.random.permutation(np.shape(imageset)[0]);
    permut = permut[:MAX_SELECTION_FROM_CATEGORY];
    fig = plt.figure()
    count = 0;
    for index in permut:
        count = count + 1;
        ax1 = fig.add_subplot(1, MAX_SELECTION_FROM_CATEGORY, count);
        ax1.set_title(chr(labelset[index]+97));
        ax1.imshow(imageset[index])


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
