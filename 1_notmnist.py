
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
from os.path import join, isfile, isdir
from os import listdir
import  exercises
from exercises import displayImagesFromTrainFolders, showImagesFromAllPickleFolders, showHistogramsForTrainTest, showImageFromDataSet

# Config the matplotlib backend as plotting inline in IPython
#get_ipython().magic('matplotlib inline')


# First, we'll download the dataset to our local machine. The data consists of characters rendered in a variety of fonts on a 28x28 image. The labels are limited to 'A' through 'J' (10 classes). The training set has about 500k and the testset 19000 labeled examples. Given these sizes, it should be possible to train models quickly on any machine.

# In[4]:

url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '.' # Change me to store data elsewhere


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

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)


# Extract the dataset from the compressed .tar.gz file.
# This should give you a set of directories, labeled A through J.

# In[5]:

num_classes = 10
np.random.seed(133)

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
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders
  
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

#PROBLEM1
#displayImagesFromTrainFolders(train_folders);
#os.system("pause");


# Now let's load the data in a more manageable format. Since, depending on your computer setup you might not be able to fit it all in memory, we'll load each class into a separate dataset, store them on disk and curate them independently. Later we'll merge them into a single dataset of manageable size.
# 
# We'll convert the entire dataset into a 3D array (image index, x, y) of floating point values, normalized to have approximately zero mean and standard deviation ~0.5 to make training easier down the road. 
# 
# A few images might not be readable, we'll just skip them.
# In[7]:

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

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

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)



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
    num_classes = len(pickle_files)
    totalImages = 0;
    for pfile in pickle_files:
        try:
          with open(pfile, 'rb') as f:
              #load data set from picke file 
              dataset = pickle.load(f);
              nDataSet = np.shape(dataset)[0];
              totalImages = totalImages + nDataSet;
        except:
              #this remained unchanged if we cannot deserialize the pickle file 
              totalImages = totalImages;

    trainSize = (int)(percentTraining*totalImages);
    validSize = totalImages - trainSize;


    #create empty datasets for training/validation set
    valid_dataset, valid_labels = make_arrays(validSize, image_size)
    train_dataset, train_labels = make_arrays(trainSize, image_size)
    entire_dataset, entire_labels = np.ndarray((totalImages, image_size, image_size), dtype=np.float32), np.ndarray(totalImages, dtype=np.int32);
    lastIndex = 0;

    #enumerate through pickle files 
    for label, pickle_file in enumerate(pickle_files):       
        try:
          with open(pickle_file, 'rb') as f:
            #load data set from pickle file 
            letter_set = pickle.load(f)
            # let's shuffle the letters to have random validation and training set
            np.random.shuffle(letter_set)
            label = exercises.getLabel(pickle_file);
            numExamples = np.shape(letter_set)[0];

            #initialize dimensions
            tr, lb = make_arrays(numExamples, image_size);

            #assign values from this dataset
            tr = letter_set; 
            lb[:numExamples] = label;

            entire_dataset[lastIndex:(lastIndex+numExamples),:,:] = tr;
            entire_labels[lastIndex:(lastIndex+numExamples)]=label;

            lastIndex = lastIndex + numExamples;

        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    permutation = np.random.permutation(totalImages);
    entire_dataset = entire_dataset[permutation,:,:];
    entire_labels = entire_labels[permutation];

    train_dataset = entire_dataset[:trainSize];
    train_labels  = entire_labels[:trainSize];
    
    if trainSize < totalImages:
        valid_dataset = entire_dataset[trainSize:];
        valid_labels =  entire_labels[trainSize:];

        return train_dataset, train_labels, valid_dataset, valid_labels
    else:
        return train_dataset, train_labels, None, None

            
            
train_dataset, train_labels, valid_dataset, valid_labels = createTrainingValidationSets(train_datasets, 0.7)
test_dataset, test_labels, _, _ = createTrainingValidationSets(test_datasets, 1)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)



# Next, we'll randomize the data. It's important to have the labels well shuffled for the training and test distributions to match.

# In[ ]:

# def randomize(dataset, labels):
#    permutation = np.random.permutation(labels.shape[0])
#    shuffled_dataset = dataset[permutation,:,:]
#    shuffled_labels = labels[permutation]
#    return shuffled_dataset, shuffled_labels
#
# #train_dataset, train_labels = randomize(train_dataset, train_labels)
# #test_dataset, test_labels = randomize(test_dataset, test_labels)
# #valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


# ---
# Problem 4
# ---------
# Convince yourself that the data is still good after shuffling!
# 
# ---

# Finally, let's save the data for later reuse:

# In[ ]:


#showImageFromDataSet(train_dataset, train_labels);
#showImageFromDataSet(test_datasets, test_labels);
#showImageFromDataSet(valid_dataset, valid_labels);
#os.system("pause");

# pickle_file = os.path.join(data_root, 'notMNIST.pickle')
#
# try:
#   f = open(pickle_file, 'wb')
#   save = {
#     'train_dataset': train_dataset,
#     'train_labels': train_labels,
#     'valid_dataset': valid_dataset,
#     'valid_labels': valid_labels,
#     'test_dataset': test_dataset,
#     'test_labels': test_labels,
#     }
#   pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
#   f.close()
# except Exception as e:
#   print('Unable to save data to', pickle_file, ':', e)
#   raise
#
# # In[ ]:
# statinfo = os.stat(pickle_file)
# print('Compressed pickle size:', statinfo.st_size)


# ---
# Problem 5
# ---------
# 
# By construction, this dataset might contain a lot of overlapping samples, including training data that's also contained in the validation and test set! Overlap between training and test can skew the results if you expect to use your model in an environment where there is never an overlap, but are actually ok if you expect to see training samples recur when you use it.
# Measure how much overlap there is between training, validation and test samples.
# 
# Optional questions:
# - What about near duplicates between datasets? (images that are almost identical)
# - Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.
# ---

#### ???? #####




# ---
# Problem 6
# ---------
# 
# Let's get an idea of what an off-the-shelf classifier can give you on this data. It's always good to check that there is something to learn, and that it's a problem that is not so trivial that a canned solution solves it.
# 
# Train a simple model on this data using 50, 100, 1000 and 5000 training samples. Hint: you can use the LogisticRegression model from sklearn.linear_model.
# 
# Optional question: train an off-the-shelf model on all the data!
# 
# ---
def flatMatrix(dataset):
    tsh = np.shape(dataset);
    reducedDataset = np.reshape(dataset, (tsh[0], tsh[1] * tsh[2]));
    return reducedDataset;

#label 0 to 9
def hotlabel(label, maxNumLabels):
    return np.eye(maxNumLabels, dtype=int)[label];

def logisticRegression(train_dataset, train_labels, test_dataset, test_labels, valid_dataset,
                       valid_labels, desiredNumberOfTrainingExamples):
    desiredNumberOfTrainingExamples = min(desiredNumberOfTrainingExamples, np.shape(train_dataset)[0]);

    reducedTrainingSet = train_dataset[:desiredNumberOfTrainingExamples,:,:];
    reducedTrainingLabels = train_labels[:desiredNumberOfTrainingExamples];

    Cs = [0.1]
    minError = 1;
    bestLogisticModel = 0;
    maxIter = 200;
    for c in Cs:
        print("fit the model for C = " + " "+ str(c));
        maxNumLabels = max(valid_labels) + 1;
        logisticRegr = LogisticRegression(max_iter = maxIter, C = c);
        logisticRegr.fit(flatMatrix(reducedTrainingSet), reducedTrainingLabels);

        flatValidationSet = flatMatrix(valid_dataset);
        predValidationSet = logisticRegr.predict_proba(flatValidationSet);

        validLabelsMatrix =  np.ndarray((valid_labels.shape[0], maxNumLabels), dtype=np.float32)
        for index in range(valid_labels.shape[0]):
            validLabelsMatrix[index] = hotlabel(valid_labels[index], maxNumLabels);

        crossEntropyError = - np.mean(np.log(predValidationSet) * validLabelsMatrix);

        if crossEntropyError < minError:
            print("Cross entropy error improved = " + " " + str(crossEntropyError));
            minError = crossEntropyError;
            bestLogisticModel = logisticRegr;

    print("Accuracy on test data" + str(bestLogisticModel.score(flatMatrix(test_dataset), test_labels)));
    return bestLogisticModel;

bestLogisticModel = logisticRegression(train_dataset, train_labels, test_dataset, test_labels, valid_dataset, valid_labels, 5000);

#a sanity check on logit regression
numChecks = 50;
for index in range(numChecks):
    print("Real class " +  str(test_labels[index]) + " predicted " + str(bestLogisticModel.predict(flatMatrix(test_dataset)[index])));
