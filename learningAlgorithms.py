from sklearn.linear_model import LogisticRegression
import numpy as np

def flatMatrix(dataset):
    tsh = np.shape(dataset);
    reducedDataset = np.reshape(dataset, (tsh[0], tsh[1] * tsh[2]));
    return reducedDataset;

#label 0 to 9
def hotlabel(label, maxNumLabels):
    return np.eye(maxNumLabels, dtype=int)[label];

# the functions assumes the dataset is organized in
def logisticRegression(train_dataset, train_labels, test_dataset, test_labels, valid_dataset,
                       valid_labels, desiredNumberOfTrainingExamples):
    desiredNumberOfTrainingExamples = min(desiredNumberOfTrainingExamples, np.shape(train_dataset)[0]);

    reducedTrainingSet = train_dataset[:desiredNumberOfTrainingExamples,:,:];
    reducedTrainingLabels = train_labels[:desiredNumberOfTrainingExamples];

    Cs = [0.1,0.3,0.5,0.9]
    bestLogisticModel = 0;
    maxIterations = [10,20,40,80,160,200,250,320,400];
    maxNumLabels = max(valid_labels) + 1;
    minError = 1;

    validLabelsMatrix = np.ndarray((valid_labels.shape[0], maxNumLabels), dtype=np.float32)
    for index in range(valid_labels.shape[0]):
        validLabelsMatrix[index] = hotlabel(valid_labels[index], maxNumLabels);

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