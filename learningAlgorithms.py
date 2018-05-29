from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import numpy as np
image_size = 28;

#unused usefull for reference
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x),axis = 0);

def flatMatrix(dataset):
    tsh = np.shape(dataset);
    reducedDataset = np.reshape(dataset, (tsh[0], tsh[1] * tsh[2]));
    return reducedDataset;

#label 0 to 9
def hotlabel(label, maxNumLabels):
    return np.eye(maxNumLabels)[label];

# datasets are 3d arrays.
def logisticRegression(train_dataset, train_labels, test_dataset, test_labels, valid_dataset,
                       valid_labels, desiredNumberOfTrainingExamples ):
    desiredNumberOfTrainingExamples = min(desiredNumberOfTrainingExamples, np.shape(train_dataset)[0]);

    reducedTrainingSet = train_dataset[:desiredNumberOfTrainingExamples,:,:];
    reducedTrainingLabels = train_labels[:desiredNumberOfTrainingExamples];

    Cs = [0.1,0.3,0.5,0.9]
    bestLogisticModel = 0;
    maxIterations = [10, 20, 40, 80, 160, 200, 250, 320, 400];
    maxNumLabels = (np.int64)(max(valid_labels) + 1);
    minError = 1;

    validLabelsMatrix = np.ndarray(shape=(valid_labels.shape[0], maxNumLabels), dtype=np.float64)
    for index in range(valid_labels.shape[0]):
        validLabelsMatrix[index] = hotlabel((np.int64)(valid_labels[index]), maxNumLabels);

    for c in Cs:
        minErrorForC = 1;
        for maxIter in maxIterations:
            print("fit the model for C = " + " "+ str(c) + " and maxIterations = " + str(maxIter));
            logisticRegr = LogisticRegression(max_iter = maxIter, C = c, solver = 'sag');
            logisticRegr.fit(flatMatrix(reducedTrainingSet), reducedTrainingLabels);

            flatValidationSet = flatMatrix(valid_dataset);
            predValidationSet = logisticRegr.predict_proba(flatValidationSet);
            crossEntropyError = - np.mean(np.log(predValidationSet) * validLabelsMatrix);

            if (crossEntropyError <= minErrorForC):
                print("Cross entropy error improved over this C= " + " " + str(crossEntropyError));
                minErrorForC = crossEntropyError;
            else:
                break;

            if crossEntropyError < minError:
                print("Cross entropy error improved globally= " + " " + str(crossEntropyError));
                minError = crossEntropyError;
                bestLogisticModel = logisticRegr;



    print("Accuracy on test data " + str(bestLogisticModel.score(flatMatrix(test_dataset), test_labels)));
    return bestLogisticModel;


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


#the datasets here are 2d arrays
def logisticRegressionWithTF(train_dataset, train_labels, test_dataset, test_labels, valid_dataset,
                       valid_labels, desiredNumberOfTrainingExamples ):

    desiredNumberOfTrainingExamples = min(desiredNumberOfTrainingExamples, np.shape(train_dataset)[0]);
    num_labels = np.shape(train_labels)[1];
    # With gradient descent training, even this much data is prohibitive.
    # Subset the training data for faster turnaround.

    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        # Load the training, validation and test data into constants that are
        # attached to the graph.
        tf_train_dataset = tf.constant(train_dataset[:desiredNumberOfTrainingExamples, :])
        tf_train_labels = tf.constant(train_labels[:desiredNumberOfTrainingExamples])
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        # These are the parameters that we are going to be training. The weight
        # matrix will be initialized using random values following a (truncated)
        # normal distribution. The biases get initialized to zero.
        weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

        # Optimizer.
        # We are going to find the minimum of this loss using gradient descent.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        # These are not part of training, but merely here so that we can report
        # accuracy figures as we train.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
    with tf.Session(graph=graph) as session:
        # This is a one-time operation which ensures the parameters get initialized as
        # we described in the graph: random weights for the matrix, zeros for the
        # biases.

        maxIterations = [10, 20, 40, 80, 160, 200, 250, 320, 400];


        tf.global_variables_initializer().run()
        minError = 1000;
        nSteps = 0;
        for step in range(400):
            # Run the computations. We tell .run() that we want to run the optimizer,
            # and get the loss value and the training predictions returned as numpy
            # arrays.
            _, l, predictions = session.run([optimizer, loss, train_prediction])
            crossEntropyError = - np.mean(np.log(valid_prediction.eval()) * valid_labels);
            print(str(crossEntropyError));

            if ( (step % 10) == 0):
                if(crossEntropyError < minError ):
                    minError = crossEntropyError;
                    nSteps = step;
                else:
                    break;

        print("Cross entropy error for steps: " + str(nSteps) + "error: " + str(minError));
        print('Test accuracy: ' + str(accuracy(test_prediction.eval(), test_labels)));