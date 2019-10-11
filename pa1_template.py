# Note: this is just a template for PA 1 and the code is for references only.
# Feel free to design the pipeline of the *main* function. However, one should keep
# the interfaces for the other functions unchanged.
# Author: Kyle D. Lee
# Date: 10-06-19

import csv
import numpy as np
from scipy import stats

def compute_accuracy(test_y, pred_y):

    # TO-DO: add your code here
    #Computing accuracy for k-nn
    

    diff = pred_y - test_y
    result = []
    for i in diff:
        if i != 0:
            result.append(1)
        else:
            result.append(0)
            
    accuracy = (diff.size - np.sum(result))/diff.size
        
    confusionMatrix = np.zeros(shape=(26,26))
        
    index = 0
    while index < len(pred_y):
        confusionMatrix[test_y[index]][pred_y[index]] += 1
        index += 1

    f_confusion_mat = open("confusionMatrix.txt", "a+")
    f_confusion_mat.write('\nConfusion Matrix for Data Set size 15000\n\n')

    value = 0
    while value < 26:
        f_confusion_mat.write('\t%s'%str(value))
        value += 1
    f_confusion_mat.write('\n')

    count = 0
    label = 0
    for element in np.nditer(confusionMatrix):
        if (count % 26 == 0):
            f_confusion_mat.write("\n%s\t"%str(label))
            label += 1
        f_confusion_mat.write(str(element))
        f_confusion_mat.write('\t')
        count += 1

    return accuracy

def countTotalPos(labelY):

    numPos = 0
    index = 0
    while index < len(labelY):
        if labelY[index] == 1:
            numPos += 1

    return numPos

def test_knn(train_x, train_y, test_x, num_nn):

    # TO-DO: add your code here
    pred_y = []
    norm2arr = []
    val = 0
    test_prd = []
    largeNum = 1000000
    verbose = False

    #print(train_x.shape)

    #iterate through each row in the test array
    for row in test_x:
        
        #Set arrays to hold value of lowest neighbors and their correspodning indecies
        k_nn_index = []
        k_nn_value = []

        #first subtract the test row from every row in train row.
        #then square each element in the new array
        #then sum everything along the rows to get norm2 squared
        arr = np.sum(np.square(train_x - row), axis=1)
        min = np.argmin(arr)

        #counter for iterations to know where to stop. Only used during testing.
        val += 1
        #norm2arr.append(list(arr))
        
        if verbose:
            print(val)
            if val == 100:
                break

        #Set current minimum incase there is no highest mode
        currentMin = arr[np.argmin(arr)]
        
        #Initialize count for iterating through the k neighbors
        count = 0
        while count < num_nn:
            count += 1
            
            #append the minimum nieghbor and its index. Then replace it with an arbitrarily high value to prevent it being selected again on second run.
            k_nn_index.append(np.argmin(arr))
            k_nn_value.append(train_y[np.argmin(arr)])
            arr[np.argmin(arr)] = largeNum
            
        #get the index of mode
        m = stats.mode(k_nn_value)

        #if a number occurs more than once and it is the greatest occurance then set that as the predicted value
        #else if the most occuring number only occurs once then append the previously set min.
        if m[1] > 1:

            result = (np.where(k_nn_value == m[0]))
            #print(k_nn_value)
            pred_y.append( k_nn_value[result[0][0]] )

        else:
            
            pred_y.append(currentMin)
        
    #Make the array nice and compatible with testY array outside
    pred_y = np.array(pred_y)

    return pred_y

def test_pocket(w, test_x):

    y_pred = []
    # TO-DO: add your code here
    for row in test_x:
        
        v = w[:,1:]
        temp_y = np.matmul(v,(row)) + w[:,0] 
        y_pred.append(np.argmax(temp_y))

    return y_pred

def train_pocket(train_x, train_y, num_iters):

    bias = float(1)                                     #Set bias as 1
    w_vector = np.array(np.random.uniform(size = 17))   #make random vector of legnth 17 with element values between 0 and 1
    w_vector[0] = bias                                  #set first value as 1
    learning_rate = 0.3                                 #Arbitrary learning rate between 0 and 1

    currIter = 0                                        #initialize current num_iters
    while currIter < num_iters:
        currRow = 0                                     #reference to the row that is being predicted.
        for row in train_x:
            
            pred = prediction(row, w_vector)            #value that is either 1 or -1 depending on what the dot product of the two vectors are.
            
            #if the predicted value matches the label then don't update anything and lower the num_iters value so we can do less tests
            if (pred > 0 and train_y[currRow] == pred) or (pred < 0 and train_y[currRow] == pred):                                
                currRow += 1                            
                currIter +=1

            #but if it doesn't update the weights to try for a better answer    
            else:                                       
                #Update the weight starting with bias since it doesn't depend on x.
                #set the index but offset it by the first term
                w_index = 1                             
                w_vector[0] = w_vector[0] + learning_rate*pred
                
                #update the rest of the weights based on perceptron algorithm
                while w_index < len(w_vector)-1:
                    w_vector[w_index] = w_vector[w_index] + learning_rate* (train_y[currRow] - pred) * row[w_index-1]
                    w_index += 1
                currRow += 1
        #Increment num_iters    
        currIter += 1
        
        #Make the w vector look nice by making it a np array
        w_vector = np.array(w_vector)

    return w_vector

def prediction(train_x, initial_w):

    #Calculate the sum of each component multiplied with other component
    #Offset by 1 for w vector since x[0] is bias term that always stays 1
    dotProd = np.dot(train_x, initial_w[1:]) + initial_w[0]

    #If greater than 0 return yes, signified as 1
    #Else return no, signified as -1
    if dotProd > 0:
        return 1
    else:
        return -1

def get_id():

    return 'tud16467'

def run_knn(data_X, data_Y, numTrainExams):
 
    train_set_size = [100, 1000, 2000, 5000, 10000]     #number of set sizes excluding 15000. 15000 will done manually
    number_of_k = [1, 3, 5, 7, 9]                       #number of k-nearest Neighbors to check for
    
    #Set size of train and test set based on numbers above
    trainX = data_X[:numTrainExams, :]              
    trainY = data_Y[:numTrainExams]
    testX = data_X[numTrainExams:, :]
    testY = data_Y[numTrainExams:]

    #Message to let user know test has started
    print( "k-nn test Start",  )

    #Open file to write results to
    fileKnn = open("knnresults.txt", "a")

    #Iterate through both set size and number of k nearest neighbors
    for setSize in train_set_size:
        print('Strating test for set size: ', setSize)
        
        for k in number_of_k:
            
            print('Strating test for k-nn: ', k)
            tempTrainX = data_X[:setSize, :]
            tempTrainY = data_Y[:setSize]
            tempTestX = data_X[20000-setSize:, :]
            tempTestY = data_Y[20000-setSize:]
            
            accu = compute_accuracy(tempTestY, test_knn(tempTrainX, tempTrainY, tempTestX, k))
            
            fileKnn.write("For Set Size: ")
            fileKnn.write(str(setSize))
            fileKnn.write(" k-value: ")
            fileKnn.write(str(k))
            fileKnn.write(" Accuracy was: ")
            fileKnn.write(str(accu))
            fileKnn.write("\n")

    for k in number_of_k:
        accu = compute_accuracy(testY, test_knn(trainX, trainY, testX, k))

        fileKnn.write("For Set Size: 15000")
        fileKnn.write(" k-value: ")
        fileKnn.write(str(k))
        fileKnn.write(" Accuracy was: ")
        fileKnn.write(str(accu))
        fileKnn.write("\n")

    fileKnn.close()

    return None

def runPerceptron(data_X, data_Y, num_iters, numTrainExams):

    train_set_size = [100, 1000, 2000, 5000, 10000] 
    numIteration = num_iters                  #arbitrarily set num iter
    lastLetter = 26
    

    #Set size of train and test set based on numbers above
    trainX = data_X[:numTrainExams, :]              
    trainY = data_Y[:numTrainExams]
    testX = data_X[numTrainExams:, :]
    testY = data_Y[numTrainExams:]

    #Message to let user know test has started
    print( 'Perceptron test Start' )

    #Open file to write results to
    filePerc = open("perceptronresults.txt", "a")

    for setSize in train_set_size:
        print('Strating perceptron train for set size: ', setSize)

        currentLabel = 0                    #initialize label to be the first letter which is 0
        w = []                              #array to hold all w vectors to corresponding labels

        tempTrainX = data_X[:setSize, :]
        tempTrainY = data_Y[:setSize]
        tempTestX = data_X[numTrainExams:, :]
        tempTestY = data_Y[numTrainExams:]

        #Calculate the weight vector for each letter
        while currentLabel < lastLetter:
            print('Starting test for set %s label %s'%(str(setSize), str(currentLabel)) )
            perc_train_y = convertForPerceptron(currentLabel, tempTrainY)
          
            v = train_pocket(tempTrainX, perc_train_y, numIteration)
            w.append(v)

            currentLabel += 1
            print(currentLabel)

        #initialize label to be the first letter which is 0

        w = np.array(w)

        y = test_pocket(w, tempTestX)
        acc = compute_accuracy(tempTestY,y)
      
        filePerc.write('For set size: ')
        filePerc.write(str(setSize))
        #filePerc.write(' \nWeightvector: ')
        #filePerc.write(str(w[currentLabel2]))
        filePerc.write('\naccuracy: ')
        filePerc.write(str(acc))
        filePerc.write('\n')

        
    filePerc.write('\n')
        

    #Set size of train and test set based on numbers above
    
    perc_train_y = convertForPerceptron(currentLabel, trainY)

    #initialize label to be the first letter which is 0
    currentLabel = 0
    w = []
    #Calculate the weight vector for each letter
    while currentLabel < lastLetter:
        print('making progress training %s'%(str(currentLabel)))
        v = train_pocket(trainX, perc_train_y, numIteration)
        w.append(v)

        currentLabel += 1
    #initialize label to be the first letter which is 0

    w = np.array(w)
    print('making progress testing writing now %s'%(str(currentLabel)))
    #return predicted values for each label
    y = test_pocket(w, testX)
    acc = compute_accuracy(testY,y)

    filePerc.write('For set size: ')
    filePerc.write(str(numTrainExams))
    #filePerc.write(' Weightvector: ')
    #filePerc.write(str(w[currentLabel]))
    filePerc.write('\naccuracy: ')
    filePerc.write(str(acc))
    filePerc.write('\n')
    currentLabel += 1

    filePerc.close()
    return None

def convertForPerceptron(label, vector):
    #function to convert label vectors to one v. all format
    i = 0
    newVector = []
    while i < len(vector):
        if vector[i] == label:
            newVector.append(1)
            i += 1
        else:
            newVector.append(-1)
            i += 1

    return newVector

def convertFortest(label, vector):
    #convert to one v. all but used to count the number of labels instead.
    #Only used for testing
    i = 0
    newVector = []
    while i < len(vector)-1:
        if vector[i] == label:
            newVector.append(1)
            i += 1
        else:
            newVector.append(0)
            i += 1

    return newVector

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

    numIteration = 10000
    
    print( "test Start",  )
    #run_knn(dataX, dataY, nNumTrainingExamples)
    runPerceptron(dataX, dataY, numIteration, nNumTrainingExamples)

    return None

if __name__ == "__main__":
    main()
