from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import numpy as np
import multiprocessing
import psutil

#unused usefull for reference
def softmax(x):
    return  np.exp(x) / np.sum(np.exp(x),axis = 0);

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

#predictions is a tensor
def accuracy(tf_Predictions, labels):
    predictions = tf_Predictions.eval();
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / np.shape(labels)[0];

#the datasets here are 2d arrays
def logisticRegressionWithTF(train_dataset, train_labels, test_dataset, test_labels, valid_dataset,
                       valid_labels, desiredNumberOfTrainingExamples, batchSizePercent):

    desiredNumberOfTrainingExamples = min(desiredNumberOfTrainingExamples, train_labels.shape[0]);
    num_labels = np.shape(train_labels)[1];
    batch_size = (np.int64)(batchSizePercent*desiredNumberOfTrainingExamples);
    # With gradient descent training, even this much data is prohibitive.
    # Subset the training data for faster turnaround.

    lambdas = [0, 0.01, 0.03, 0.09, 0.3, 0.9, 3, 9]
    globalError = -1;
    # m*n 2d matrix
    testPredictions = np.array([]);
    #scalar, test set classification accuracy.
    accRes = 0 ;
    image_size = (int)(np.sqrt(np.shape(train_dataset)[1]));

    graph = tf.Graph()
    for lambdaV in lambdas:
        with graph.as_default():
            # Input data.
            # Load the training, validation and test data into constants that are
            # attached to the graph.

            tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size));
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels));
            tf_valid_dataset = tf.constant(valid_dataset);
            tf_test_dataset = tf.constant(test_dataset);
            tf_lambda = tf.constant(lambdaV);

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
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) \
                   + tf.multiply(tf.reduce_sum(tf.square(weights)), tf_lambda);

            # Optimizer.
            # We are going to find the minimum of this loss using gradient descent.
            optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

            # Predictions for the training, validation, and test data.
            # These are not part of training, but merely here so that we can report
            # accuracy figures as we train.
            tf_TrainPrediction = tf.nn.softmax(logits)
            valid_logits = tf.matmul(tf_valid_dataset, weights) + biases;
            tf_testPrediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
        with tf.Session(graph=graph) as session:
            # This is a one-time operation which ensures the parameters get initialized as
            # we described in the graph: random weights for the matrix, zeros for the
            # biases.

            tf.global_variables_initializer().run();

            minError = -1;
            for step in range(400):
                # Run the computations. We tell .run() that we want to run the optimizer,
                # and get the loss value and the training predictions returned as numpy
                # arrays.

                offset = (step * batch_size) % (desiredNumberOfTrainingExamples - batch_size)

                batch_data = train_dataset[offset:(offset + batch_size), :]
                batch_labels = train_labels[offset:(offset + batch_size), :]
                fd = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

                _, l, predictions = session.run([optimizer, loss, tf_TrainPrediction], feed_dict=fd);

                crossEntropyError = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=valid_labels, logits=valid_logits.eval()));
                ce =  crossEntropyError.eval();

                if minError < 0:
                    minError = ce;
                if globalError < 0:
                    globalError = ce;

                if(ce < minError ):
                    print("New minimum error found " + str(ce) + " steps " + str(step) + " lambda-reg " + str(lambdaV));
                    minError = ce;
                else:
                    break;

                if(ce < globalError):
                    print("New minimum global error found " + str(ce) + " steps " + str(step) + " lambda-reg " + str(lambdaV));
                    globalError = ce;
                    accRes = accuracy(tf_testPrediction, test_labels);
                    print('New test accuracy: ' + str(accRes));

    return testPredictions, accRes;


#the datasets here are 2d arrays
#implements two layers deep nn

#data is 2d array
def truncate(data, maxSize):
    desiredSize = min(np.shape(data)[0], maxSize);
    if data.ndim > 1:
        return data[:desiredSize, :];
    return  data[:desiredSize];


def nnWithTF(train_dataset, train_labels, test_dataset, test_labels, valid_dataset,
             valid_labels, desiredNumberOfValidationExamples, batch_size, useRegularization = False, useCovNet = True, useDropOut = True, numChannels = 1):

    logicalProcessors = multiprocessing.cpu_count();
    image_size = (int)(np.sqrt(np.shape(train_dataset)[1]));
    numTrainingExamples = np.shape(train_labels)[0];

    cluster = tf.train.ClusterSpec({
        "worker": [
            "localhost:2222",
            "localhost:2223",
            "localhost:2224"
        ],
        "ps": [
            "localhost:2225",
            "localhost:2226"
        ]})

    serverWorker1 = tf.train.Server(cluster, job_name="worker", task_index=0)
    serverWorker2 = tf.train.Server(cluster, job_name="worker", task_index=1)
    serverWorker3 = tf.train.Server(cluster, job_name="worker", task_index=2)

    serverPs1 = tf.train.Server(cluster, job_name="ps", task_index=0)
    serverPs2 = tf.train.Server(cluster, job_name="ps", task_index=1)

    valid_dataset = truncate(valid_dataset, desiredNumberOfValidationExamples);
    valid_labels = truncate(valid_labels, desiredNumberOfValidationExamples);

    test_dataset = truncate(test_dataset, desiredNumberOfValidationExamples);
    test_labels = truncate(test_labels, desiredNumberOfValidationExamples);

    num_labels = np.shape(train_labels)[1];
    # With gradient descent training, even this much data is prohibitive.
    # Subset the training data for faster turnaround.

    globalError = -1;
    # m*n 2d matrix
    testPredictions = np.array([]);
    #scalar, test set classification accuracy.
    accRes = 0 ;

    #covnets use cubes
    if useCovNet:
        train_dataset = train_dataset.reshape((-1, image_size, image_size, numChannels)).astype(np.float32)
        test_dataset = test_dataset.reshape((-1, image_size, image_size, numChannels)).astype(np.float32)
        valid_dataset = valid_dataset.reshape((-1, image_size, image_size, numChannels)).astype(np.float32)

    graph = tf.Graph();

    #regularization term
    lambdaV = 0.1;

    with graph.as_default():

        tf_lambda = tf.constant(lambdaV, dtype=np.float32);

        with tf.device("/job:ps/task:0/cpu:0"):
            maxNumSteps = 200;
            if not useCovNet:
                tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size));
            else:
                tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, numChannels))

            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels));

            #used for compute crossEntropy error in learning
            tf_valid_dataset = tf.constant(valid_dataset);
            tf_valid_labels = tf.constant(valid_labels);

            #used to compute predictions and compute accuracy during learning, accuracy uses
            #numpy arrays so not necessary to use tensor for labels
            tf_test_dataset = tf.constant(test_dataset);

        def computeRequiredMemoryAndNumberofParameters():
            if useCovNet:
                # for hiddenLayerWidth in [4, 8, 16, 32, 64, 128, 256]:
                #     for covNetDepth in [2, 4, 8, 16, 32, 64]:
                #         nparams = patchSize ** 2 * covNetDepth * (numChannels + covNetDepth) + 2 * covNetDepth
                #         nparams = nparams + hiddenLayerWidth * (hiddenLayerWidth + num_labels + 2 + image_size * image_size / 8) + num_labels;
                #         if (nparams > numTrainingExamples or (
                #                 nparams * 4) > psutil.virtual_memory().available * 0.5):
                #             return covNetDepth, hiddenLayerWidth;
                return 64, 512;
            else:
                solutions = np.roots(np.array([1, image_size * image_size + num_labels + 2,
                                               num_labels - numTrainingExamples]));
                hiddenLayerWidth = 1;
                if np.isreal(solutions[0]) and solutions[0] > 0:
                    hiddenLayerWidth = np.ceil(solutions[0]);
                else:
                    hiddenLayerWidth = np.ceil(solutions[1]);

                hiddenLayerWidth = (int)(hiddenLayerWidth);

                for hlw in range(hiddenLayerWidth, hiddenLayerWidth + 99999):
                    if  0 == (hlw & (hlw - 1)):
                        return 0, hlw


        # Variables.
        # These are the parameters that we are going to be training. The weight
        # matrix will be initialized using random values following a (truncated)
        # normal distribution. The biases get initialized to zero.
        with tf.device("/job:ps/cpu:0"):

            if not useCovNet:
                covNetDepth = 1
                covNetLayers = 0
                _, hiddenLayerWidth = computeRequiredMemoryAndNumberofParameters();

            if useCovNet:
                patchSize = 8
                covNetLayers = 2

                covNetDepth,hiddenLayerWidth =  computeRequiredMemoryAndNumberofParameters();


                weights1CV = tf.Variable(tf.truncated_normal([patchSize, patchSize, numChannels, covNetDepth]))
                biases1CV = tf.Variable(tf.zeros([covNetDepth]))
                weights2CV = tf.Variable(tf.truncated_normal([patchSize, patchSize, covNetDepth, covNetDepth]))
                biases2CV = tf.Variable(tf.zeros([covNetDepth]))


            weights1 = tf.Variable(tf.truncated_normal([(int)((image_size * image_size * covNetDepth)/(4**covNetLayers)), hiddenLayerWidth]));
            weights2 = tf.Variable(tf.truncated_normal([hiddenLayerWidth, hiddenLayerWidth]));
            weights3 = tf.Variable(tf.truncated_normal([hiddenLayerWidth, num_labels]));

            biases1 = tf.Variable(tf.zeros([hiddenLayerWidth]));
            biases2 = tf.Variable(tf.zeros([hiddenLayerWidth]));
            biases3 = tf.Variable(tf.zeros([num_labels]));

        # Training computation.
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
        # input is a tf tensor
        def model(data, useCovNet=True, useDropOut = True):

            reshape = data
            if useCovNet:
                conv = tf.nn.conv2d(data, weights1CV, [1, 1, 1, 1], padding='SAME')
                hidden = tf.nn.leaky_relu(conv + biases1CV)
                hidden = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

                conv = tf.nn.conv2d(hidden, weights2CV, [1, 1, 1, 1], padding='SAME')
                hidden = tf.nn.leaky_relu(conv + biases2CV)
                hidden = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

                shape = hidden.get_shape().as_list()
                reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

            y1 = tf.matmul(reshape, weights1) + biases1
            y1 = tf.nn.leaky_relu(y1);
            if useDropOut:
                y1 = tf.nn.dropout(y1, 0.8);

            y2 = tf.matmul(y1, weights2) + biases2;
            y2 = tf.nn.leaky_relu(y2);
            if useDropOut:
                y2 = tf.nn.dropout(y2, 0.8);

            return  tf.matmul(y2, weights3) + biases3;




        with tf.device("/job:worker/task:0/gpu:0"):
            y3 = model(tf_train_dataset, useDropOut=useDropOut, useCovNet=useCovNet)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=y3));

        with tf.device("/job:worker/task:1/cpu:0"):
            if useRegularization:
                t1 = tf.multiply(tf.reduce_sum(tf.square(weights1)), tf_lambda);
                t2 = tf.multiply(tf.reduce_sum(tf.square(weights2)), tf_lambda);
                t3 = tf.multiply(tf.reduce_sum(tf.square(weights3)), tf_lambda);
                loss = loss + t1 + t2 + t3;


            # Optimizer.
            # We are going to find the minimum of this loss using gradient descent.
            # learning_rate = tf.train.exponential_decay(learning_rate = 0.6, global_step = global_step, decay_steps=maxNumSteps,decay_rate=0.999, staircase=True);
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step);
            optimizer = tf.train.AdamOptimizer(0.01).minimize(loss);

            # Predictions for the training, validation, and test data.
            # These are not part of training, but merely here so that we can report
            # accuracy figures as we train.
            train_prediction = tf.nn.softmax(y3);

        with tf.device("/job:worker/task:2"):
            with tf.device("/gpu:0"):
                logitsV  = model(tf_valid_dataset, useDropOut = False, useCovNet = useCovNet);
                tf_ValidPrediction = tf.nn.softmax(logitsV);
                #used to calculate accuracy in learning
                tf_TestPrediction = tf.nn.softmax(model(tf_test_dataset, useDropOut = False, useCovNet = useCovNet));


    with tf.Session(graph=graph, target=serverPs1.target, config = tf.ConfigProto(intra_op_parallelism_threads=logicalProcessors*4, inter_op_parallelism_threads=logicalProcessors*4, \
                    allow_soft_placement=True)) as session:
        # This is a one-time operation which ensures the parameters get initialized as
        # we described in the graph: random weights for the matrix, zeros for the
        # biases.

        tf.global_variables_initializer().run();

        lastNerror = -1;

        for global_step in range(maxNumSteps):
            # Run the computations. We tell .run() that we want to run the optimizer,
            # and get the loss value and the training predictions returned as numpy
            # arrays.


            offset = global_step * batch_size;
            if( offset >= numTrainingExamples ):
                offset = 0;


            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            fd = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=fd);

            crossEntropyError = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_valid_labels, logits=logitsV))).eval();

            print("Error found " + str(crossEntropyError) + " steps " + str(global_step) + " globalError " + str(globalError));


            if lastNerror < 0:
                lastNerror = crossEntropyError;

            if global_step % 5 == 0:
                if lastNerror < crossEntropyError:
                    break;
                else:
                    lastNerror = crossEntropyError;

            if globalError < 0:
                globalError = crossEntropyError;

            if crossEntropyError < globalError:
                print("New minimum global error found " + str(crossEntropyError) + " steps " + str(global_step) + " lambda-reg " + str(lambdaV));
                globalError = crossEntropyError;
                accRes = accuracy(tf_TestPrediction, test_labels);
                print('New test accuracy: ' + str(accRes));

    return testPredictions, accRes;


