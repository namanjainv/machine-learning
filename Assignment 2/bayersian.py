import numpy as np
from copy import deepcopy

filename = "Dataset/table4Train.txt"
filename2 = "Dataset/table4Test.txt"
class_attribute = 2

# Prepare the file to a complete numpy array
def prepareFileData(filename, delimiter = " "):
    fileData = None
    f = open(filename, "r")
    for line in f:
        line = line.strip().split(delimiter)
        x = len(line)
        dataPoint = np.array( line )
        dataPoint = np.reshape( dataPoint, newshape = ( -1, x ) )
        if fileData is None:
            fileData = dataPoint
        else: 
            fileData = np.append( fileData, dataPoint, axis = 0 )
    return fileData

def findUniqueValues( array ):
    unique = []
    for item in array:
        if item not in unique:
            unique.append(item)
    return unique

def findUniqueAttributes( fileData, classAttributeNumber ):

    prior = { }
    posterior = { }
    likelihood = { }
    N, attributes_count = np.shape( fileData )

    # Initialize prior and posterior
    for attribute_index in range( attributes_count ):
        attribute_unique_values = findUniqueValues( fileData[:, attribute_index] )
        attribute_object = { }
        for attribute_value in attribute_unique_values:
            attribute_object[attribute_value] = 0
            if( attribute_index == classAttributeNumber ):
                posterior[ attribute_value ] = 0
        if( attribute_index != classAttributeNumber ):
            prior[ attribute_index ] = attribute_object

    # Initialize Likelihood
    for className in posterior:
        likelihood[ className ] = deepcopy( prior )

    # Finding count for prior, posterior and likelihood
    for trainData_index in range( N ):
        trainData = fileData[ trainData_index, : ]
        myClass = trainData[ classAttributeNumber ]
        posterior[ myClass ] += 1
        for attribute_index in range( attributes_count ):
            if( attribute_index != classAttributeNumber ):
                prior[ attribute_index ][ trainData[attribute_index] ] += 1
                likelihood[ myClass ][ attribute_index ][ trainData[ attribute_index ] ] += 1

    print( "Count: " )
    print( "Size: " + str(N) )
    print( "Prior: " + str(prior) )
    print( "Posterior: " + str(posterior) )
    print( "Likelihood: " + str(likelihood) )
    print( "==================================" )

    # Finding likelihood probabilites
    for className in likelihood:
        myAttributeCountObject = likelihood[ className ]
        for attribute_index in range( attributes_count ):
            if attribute_index != classAttributeNumber:
                for attribute_value in prior[ attribute_index ]:
                    myAttributeCountObject[ attribute_index ][ attribute_value ] /= posterior[ className ]


    # Finding Probablities for prior
    for attribute_index in range( attributes_count ):
        if attribute_index != classAttributeNumber:
            for attribute_value in prior[ attribute_index ]:
                prior[ attribute_index ][ attribute_value ] /= N
        else:
            for attribute_value in posterior:
                posterior[ attribute_value ] /= N

    print( "Probability: " )
    print( "Prior: " + str(prior) )
    print( "Posterior: " + str(posterior) )
    print( "Likelihood: " + str(likelihood) )
    print( "==================================" )

    return prior, posterior, likelihood

def predict( testData, prior, posterior, likelihood, classAttributeNumber ):

    def findJointProbability(myData, probabilityChart):
        print(probabilityChart)
        p = 1
        for attribute_index in range( attributes_count ):
            if( attribute_index != classAttributeNumber ):
                attribute_value = myData[ attribute_index ]
                p *= probabilityChart[ attribute_index ][ attribute_value ]
        return p

    (N, attributes_count) = np.shape(testData)
    for testData_index in range( N ):
        # p = { }
        # print( findJointProbability( testData[testData_index, :], prior ) )
        myPrior = findJointProbability( testData[testData_index, :], prior )
        print("Data: " + str( testData[testData_index, :] ))
        # print("Prior: " + str( myPrior ))
        for className in posterior:
            myPosterior = posterior[ className ]
            myLikelihood = findJointProbability( testData[testData_index, :], likelihood[ className ] )
            p = myLikelihood * myPosterior / myPrior 
            print("Class: " + str( className ) )
            print("Probability: " + str( p ) )
        print("---------------")
        # print("My class: " + str( testData[ testData_index, classAttributeNumber ] ) )
    return 1

trainData = prepareFileData(filename, delimiter = " ")
# print(fileData)
prior, posterior, likelihood = findUniqueAttributes( trainData, classAttributeNumber = class_attribute )
testData = prepareFileData(filename2, delimiter = " ")
predict( testData, prior = prior, posterior = posterior, likelihood = likelihood, classAttributeNumber = class_attribute )

