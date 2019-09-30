# Note: this is just a template for PA 1 and the code is for references only.
# Feel free to design the pipeline of the *main* function. However, one should keep
# the interfaces for the other functions unchanged.

import csv
import numpy as np

def compute_accuracy(test_y, pred_y):

    # TO-DO: add your code here

    return None

def test_knn(train_x, train_y, test_x, num_nn):

    # TO-DO: add your code here

    return None

def test_pocket(w, test_x):

    # TO-DO: add your code here

    return None

def train_pocket(train_x, train_y, num_iters):

    # TO-DO: add your code here

    return None

def get_id():

    # TO-DO: add your code here

    return 'tud16467'

def main():

    # Read the data file
    szDatasetPath = './letter-recognition.data' # Put this file in the same place as this script
    listClasses = []
    listAttrs = []
    with open(szDatasetPath) as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        for row in csvReader:
            listClasses.append(row[0])
            listAttrs.append(list(map(float, row[1:])))

    # Generate the mapping from class name to integer IDs
    mapCls2Int = dict([(y, x) for x, y in enumerate(sorted(set(listClasses)))])

    # Store the dataset with numpy array
    dataX = np.array(listAttrs)
    dataY = np.array([mapCls2Int[cls] for cls in listClasses])


    # Split the dataset as the training set and test set
    nNumTrainingExamples = 15000
    trainX = dataX[:nNumTrainingExamples, :]
    trainY = dataY[:nNumTrainingExamples]
    testX = dataX[nNumTrainingExamples:, :]
    testY = dataY[nNumTrainingExamples:]

    # TO-DO: add your code here
    num_train = 100       #Number of training examples
    num_test = 50        #Number of testing examples
    num_dims = 16        #The dimensionality of the examples 16 total

    test = dataX[nNumTrainingExamples]

    print(test)

    """
    For k-nn 
        where train_x is a (num_train, num_dims) data matrix, test_x is a (num_test, num_dims) data matrix, 
        train_y is a (num_train,) label vector, and pred_y is a (num_test,) label vector, 
        and num_nn is the number of nearest neighbors for classification.
    
    For train_Pocket
        where train_x is a (num_train, num_dims) data matrix, train_y is a (num_train,) +1/-1 label vector, 
        num_iters is the number of iterations for the algorithm, w is a vector of learned perceptron weights.

    For test_Pockets
        where w is a vector of learned perceptron weights, test_x is a (num_test, num_dims) data matrix, 
        and pred_y is a (num_test,) +1/-1 label vector.

    For acc
        where test_y is a (num_test,) label vector, and pred_y is a (num_test,) 
        label vector, and acc is a float between 0.0 and 1.0, representing the classification accuracy.
    """

    return None

if __name__ == "__main__":
    main()
