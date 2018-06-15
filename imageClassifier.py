
# coding: utf-8

# Deep Learning
# =============
# 
# Assignment 1
# ------------
# 
# The objective of this assignment is to learn about simple data curation practices, and familiarize you with some of the data we'll be reusing later.
# 
# This notebook uses the [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) dataset to be used with python experiments. This dataset is designed to look like the classic [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, while looking a little more like real data: it's a harder task, and the data is a lot less 'clean' than MNIST.

# In[3]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
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
import  visualisations
import learningAlgorithms

from sklearn import preprocessing

from visualisations import displayImagesFromTrainFolders, showImagesFromAllPickleFolders, showHistogramsForTrainTest, showImageFromDataSet
from sklearn.metrics.pairwise import laplacian_kernel



# Config the matplotlib backend as plotting inline in IPython
#get_ipython().magic('matplotlib inline')


# First, we'll download the dataset to our local machine. The data consists of characters rendered in a variety of fonts on a 28x28 image. The labels are limited to 'A' through 'J' (10 classes). The training set has about 500k and the testset 19000 labeled examples. Given these sizes, it should be possible to train models quickly on any machine.

# In[4]:

url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '.' # Change me to store data elsewhere
np.random.seed(133)
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  dest_filename = os.path.join(data_root, filename)
  if force or not os.path.exists(dest_filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(dest_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', dest_filename)
  else:
    raise Exception(
      'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
  return dest_filename




# Extract the dataset from the compressed .tar.gz file.
# This should give you a set of directories, labeled A through J.

# In[5]:




def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall(data_root)
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]

  print(data_folders)
  return data_folders

#PROBLEM1
#displayImagesFromTrainFolders(train_folders);
#os.system("pause");


# Now let's load the data in a more manageable format. Since, depending on your computer setup you might not be able to fit it all in memory, we'll load each class into a separate dataset, store them on disk and curate them independently. Later we'll merge them into a single dataset of manageable size.
# 
# We'll convert the entire dataset into a 3D array (image index, x, y) of floating point values, normalized to have approximately zero mean and standard deviation ~0.5 to make training easier down the road. 
# 
# A few images might not be readable, we'll just skip them.
# In[7]:



#return the data set associated with that letter.
def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (imageio.imread(image_file).astype(float) -
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except (IOError, ValueError) as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset

#returns a list of the pickles files coresponding to data_folders
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
      
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

#Problem2
#showImagesFromAllPickleFolders(train_datasets, test_datasets);
#os.system("pause");


           

# ---
# Problem 3
# ---------
# Another check: we expect the data to be balanced across classes. Verify that.
#
# ---
# In[29]:
#print(exercises.getLabel(".\\notMNIST_large\F.pickle"));
#showHistogramsForTrainTest(train_datasets, test_datasets);
#os.system("pause");


# Merge and prune the training data as needed. Depending on your computer setup, you might not be able to fit it all in memory, and you can tune `train_size` as needed. The labels will be stored into a separate array of integers 0 through 9.
# 
# Also create a validation dataset for hyperparameter tuning.

# In[31]:

# create 2  empty dataset one with image data, one with label data
def make_arrays(nb_rows, img_size):
  dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
  labels = np.ndarray(nb_rows, dtype=np.int32)  
  return dataset, labels

#create a cross validation set having the size of valid_size , half of the trainign set.
def createTrainingValidationSets(pickle_files, percentTraining):

    #create empty datasets for training/validation set
    valid_dataset, valid_labels = np.array([]), np.array([]);
    train_dataset, train_labels = np.array([]), np.array([]);

    valid_dataset.shape = (0, image_size, image_size);
    train_dataset.shape = (0, image_size, image_size);

    valid_labels.shape = 0;
    train_labels.shape = 0;

    # valid_dataset.dtype = np.float32;
    # valid_labels.dtype  = np.int32;
    #
    # train_dataset.dtype = np.float32;
    # train_labels.dtype = np.int32;


    #enumerate through pickle files
    for pickle_file in pickle_files:
        try:
          with open(pickle_file, 'rb') as f:
            #load data set from pickle file 
            letter_set = pickle.load(f)
            # let's shuffle the letters to have random validation and training set
            np.random.shuffle(letter_set)
            label = visualisations.getLabel(pickle_file);
            numExamples = np.shape(letter_set)[0];

            trainSize = (int)(percentTraining * numExamples);
            validSize = numExamples - trainSize;

            #initialize dimensions
            training,   labelsTraining = make_arrays(trainSize, image_size);
            validation, labelsValidation = make_arrays(validSize, image_size);

            #assign values from this dataset
            training = letter_set[:trainSize, :, :];
            labelsTraining[:trainSize] = label;

            validation = letter_set[trainSize:, :, :];
            labelsValidation[:validSize] = label;

            valid_dataset = np.append(valid_dataset, validation, axis=0);
            valid_labels = np.append(valid_labels, labelsValidation, axis=0);

            train_dataset = np.append(train_dataset, training, axis=0);
            train_labels  = np.append(train_labels, labelsTraining, axis=0);


        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    permutationTrain = np.random.permutation(np.shape(train_labels)[0]);
    permutationValid = np.random.permutation(np.shape(valid_labels)[0]);

    train_dataset = train_dataset[permutationTrain, :, :];
    train_labels = train_labels[permutationTrain];

    valid_dataset = valid_dataset[permutationValid, :, :];
    valid_labels  = valid_labels[permutationValid];

    return train_dataset, train_labels, valid_dataset, valid_labels


def reformatForTensorFlow(dataset, labels, maxNumLabels):
  datasetFormated = learningAlgorithms.flatMatrix(dataset);
  datasetFormated = datasetFormated.astype(np.float32);
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  validLabelsMatrix = np.ndarray(shape = (labels.shape[0], maxNumLabels), dtype=np.int32);
  for index in range(labels.shape[0]):
      validLabelsMatrix[index] = learningAlgorithms.hotlabel((np.int64)(labels[index]), maxNumLabels);
  return datasetFormated, validLabelsMatrix


pickle_file = 'notMNIST.pickle'
dest_filename = os.path.join(data_root, pickle_file)

#load the datasets from pickle file , if the pickle file exists
if os.path.exists(dest_filename):
    with open(pickle_file, 'rb') as f:
      save = pickle.load(f)
      train_dataset = save['train_dataset']
      train_labels = save['train_labels']
      valid_dataset = save['valid_dataset']
      valid_labels = save['valid_labels']
      test_dataset = save['test_dataset']
      test_labels = save['test_labels']
      del save  # hint to help gc free up memory
      print('Training set', train_dataset.shape, train_labels.shape)
      print('Validation set', valid_dataset.shape, valid_labels.shape)
      print('Test set', test_dataset.shape, test_labels.shape)

else:
    train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
    test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
    train_folders = maybe_extract(train_filename)
    test_folders = maybe_extract(test_filename)

    #these are pickle files
    # shape is m x 28 x 28
    train_datasets = maybe_pickle(train_folders, 45000)
    test_datasets = maybe_pickle(test_folders, 1800)

    #these are nparrays holding the data, non label datasets are 3d arrays, label data is 1d array
    train_dataset, train_labels, valid_dataset, valid_labels = createTrainingValidationSets(train_datasets, 0.7)
    test_dataset, test_labels, _, _ = createTrainingValidationSets(test_datasets, 1)


    print('Training:', train_dataset.shape, train_labels.shape)
    print('Validation:', valid_dataset.shape, valid_labels.shape)
    print('Testing:', test_dataset.shape, test_labels.shape)

# visualisations.showImageFromDataSet(train_dataset, train_labels, 3);
# visualisations.showImageFromDataSet(test_dataset, test_labels, 3);

#returns hot labels for labels, and 3d array into 2d array for non label data.
# for label data returns a hot label 2d matrix
numLabels = (np.int64)(max(max(train_labels), max(valid_labels)) + 1);
train_dataset, train_labels = reformatForTensorFlow(train_dataset, train_labels, numLabels)
valid_dataset, valid_labels = reformatForTensorFlow(valid_dataset, valid_labels, numLabels)
test_dataset, test_labels  = reformatForTensorFlow(test_dataset, test_labels, numLabels)


# for nc in [8, 16,32,64,128,144,196,256,289, 400, 576, 676, 784]:



print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# nFeatures = np.shape(train_dataset)[1];
# permut = np.random.permutation(np.shape(train_dataset)[0])
# landmarks = train_dataset[permut[:nFeatures], :];
#
# train_dataset = laplacian_kernel(train_dataset, landmarks);
# valid_dataset = laplacian_kernel(valid_dataset, landmarks);
# test_dataset = laplacian_kernel(valid_dataset, landmarks);





# _,_ = learningAlgorithms.logisticRegressionWithTF(train_dataset, train_labels, test_dataset, test_labels, valid_dataset, valid_labels, 5000, 0.1);
_,_ = learningAlgorithms.nnWithTF(train_dataset, train_labels, test_dataset, test_labels, valid_dataset, valid_labels, 512, 64,
                                  useRegularization=False, useDropOut=True, useCovNet=True, usePCA= True);