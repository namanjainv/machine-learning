# -*- coding: utf-8 -*-
"""ML Assignment 2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GomaSSsEdOy5TCduTfLlQTyYa5-7vWX2
"""

from google.colab import drive
drive.mount('/content/drive')

from google.colab import files
from IPython.display import HTML, display

import numpy as np
import io

# REQUIRED
datasetFileName = 'Assignment2_dataset.txt'
classAttributeIndex = 8

# PARAMETERS
dataSplitRatio = 0.2

# Function to read a file
def readFile( fileName ):
  with open( fileName, 'r' ) as f:
    lines = f.read().split( '\n' )
    return lines

print("#### FILE DATA ####")
fileData = readFile( datasetFileName )
for line in fileData:
  print( line )

# Converting the file data into a 2D array
def tabulateData( data, delimiter = ' ', hasHeader = True ):
  X = []
  for line in data:
    words = line.split(delimiter)
    X.append(words)
  return X

print("#### TABULATED DATA ####")
tabulatedData = tabulateData( fileData )
display(HTML(
   '<table><tr>{}</tr></table>'.format(
       '</tr><tr>'.join(
           '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in tabulatedData)
       )
))

# Removing data points which consists of null values
def preprocessData( tabulatedData ):
  X = []
  requiredLength = len( tabulatedData[0] )
  for dataPoint in tabulatedData:
    if( len(dataPoint) < requiredLength ):
      continue
    # if "none" in dataPoint:
    #   continue
    X.append( dataPoint[ :requiredLength ] )
  
  return X

print("#### PREPROCESSED DATA ####")
preprocessedData = preprocessData( tabulatedData )
display(HTML(
   '<table><tr>{}</tr></table>'.format(
       '</tr><tr>'.join(
           '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in preprocessedData)
       )
))

# Function to split data into Training and Testing
def processData( preprocessedData, classAttributeIndex = classAttributeIndex, split = dataSplitRatio ):

  # Converting to a numpy array
  headers = preprocessedData[0]
  preprocessedData = np.asarray( preprocessedData[1:] )
  N, M = np.shape( preprocessedData )

  X_test = np.empty( shape = (1, M - 1) )
  Y_test = np.empty( shape = (1, 1) )

  # Converting to respective train and test set
  Y = preprocessedData[ : , classAttributeIndex ].reshape(-1,1)
  X = np.delete( preprocessedData, classAttributeIndex, axis = 1 )

  # Finding test data based on split
  i = 0
  while( True ):
    if( i/N >= dataSplitRatio ):
      break
    i += 1
    random_index = np.random.randint( np.shape(X)[0] )

    X_test = np.append(X_test, np.matrix( X[random_index, :] ), axis = 0)
    Y_test = np.append(Y_test, np.matrix( [ Y[random_index] ] ), axis = 0)
    
    X = np.delete( X, random_index, axis = 0 )
    Y = np.delete( Y, random_index, axis = 0 )

  X_test = np.delete( X_test, 0, axis = 0 )
  Y_test = np.delete( Y_test, 0, axis = 0 )

  return X, Y, X_test, Y_test, headers

X_train, Y_train, X_test, Y_test, headers = processData(preprocessedData)

print("#### TRAINING DATA ####")
print(X_train)
print(Y_train)
print()
print("#### TESTING DATA ####")
print(X_test)
print(Y_test)

def knn( X_train, Y_train, X_test, Y_test, k = 3 ):

  def distance(ptA, ptB):
    diff = ( ptA == ptB )
    return np.size(diff) - np.sum(diff)

  def getKNNeighbours(X_train, Y_train, testPoint, k):
    dist = np.empty((1,3))

    # Finding distance with all the training nodes and storing in dist matrix 
    N_train, M = np.shape(X_train)

    for j in range( N_train ):
      temp = np.array([[ j, distance( X_train[j, :], testPoint ), Y_train[j, 0] ]])
      dist = np.append( dist, temp, axis = 0 )
    dist = np.delete(dist, 0, axis = 0)

    # Sorting the distances
    dist = dist[dist[:, 1].argsort()] 

    # Selecting top K elements as our neighbours
    neighbours = dist[:k, :]

    return neighbours

  def predict( neighbours ):
    # Finding count class
    countClass = dict()
    for neighbourClass in neighbours[:, 2]:
      if( neighbourClass in countClass ):
        countClass[neighbourClass] += 1
      else:
        countClass[neighbourClass] = 1

    # Finding prediction
    maxCount = -1
    predClass = None
    for countClassKey in countClass:
      if( maxCount < countClass[ countClassKey ] ):
        maxCount = countClass[ countClassKey ]
        predClass = countClassKey

    return predClass

  truePredicted = 0

  N_test, M = np.shape(X_test)
  for i in range( N_test ):
    dist = np.empty((1,3))
    print("Datapoint: " + str( X_test[i, :] ) )
    print("Actual Class: " + str( Y_test[i, 0] ) )

    neighbours = getKNNeighbours( X_train, Y_train, X_test[i, :], k = k )
    predClass = predict( neighbours )

    print("Predicted Class: " + predClass)
    print("--------------------")

    if( predClass == Y_test[ i, 0 ] ):
      truePredicted += 1
      
  print("Accuracy: " + str( truePredicted*100.0/N_test ) )


print('---------------------------- KNN ------------------------------------')
knn( X_train, Y_train, X_test, Y_test )

del headers[ classAttributeIndex ]

def bayersian( X_train, Y_train, X_test, Y_test, headers, classAttributeIndex ):

  def findProbability(data):
    countClass = dict()
    for value in data:
      if value in countClass:
        countClass[value] += 1
      else:
        countClass[value] = 1

    N = len(data)
    probability = dict()
    for key in countClass:
      probability[key] = countClass[key]/N

    return probability

  def fit( X_train, Y_train, X_test, Y_test, headers, classAttributeIndex ):
    Y_prior = findProbability(Y_train[:,0])

    N_train, M = np.shape(X_train)

    print("Prior Probabilities: ")
    print( Y_prior )
    print( )

    conditional = dict( )
    for className in Y_prior:
      conditional[className] = dict()
      classDataIndex = np.where(Y_train == className)[0]
      i = 0
      for attribute in range( M ):
        conditional[ className ][ headers[attribute] ] = findProbability( X_train[classDataIndex, attribute] )

    print("Conditional Probabilities: ")
    print( conditional )
    print( )
  
    return Y_prior, conditional

  def predict(Y_prior, conditional, testPoint, headers, classAttributeIndex):

    def jointProbability( testPoint, probabilities, headers ):
      mul = 1
      M = len(headers)
      for i in range(M):
        headerProbabilityValues = probabilities[ headers[i] ]
        if not testPoint[0,i] in headerProbabilityValues:
          return 0
        mul *= headerProbabilityValues[ testPoint[0,i]  ]
      return mul

    def findClass( probability ):
      maxProbability = -1
      maxClassName = None
      for className in probability:
        if( probability[className] > maxProbability ):
          maxProbability = probability[className]
          maxClassName = className
      return maxClassName

    probability = dict()
    for className in Y_prior:
      num2 = Y_prior[className]

      num1 = jointProbability( testPoint, conditional[className], headers )

      denom = 1
      for denoClassName in Y_prior:
        denom += jointProbability( testPoint, conditional[denoClassName], headers )*Y_prior[denoClassName]
      probability[className] = num1*num2/denom
    
    print(probability)
    return findClass(probability)

  Y_prior, conditional = fit( X_train, Y_train, X_test, Y_test, headers, classAttributeIndex )

  N_test, M = np.shape( X_test )
  truePredicted = 0
  for i in range( N_test ):
    print("TestPoint: " + str(X_test[i,:]))
    print("Actual Class: " + str( Y_test[i, 0] ) )

    predClass = predict( Y_prior, conditional, X_test[i,:], headers, classAttributeIndex )

    print("Predicted Class: " + predClass)
    print("--------------------")

    if( predClass == Y_test[ i, 0 ] ):
      truePredicted += 1
      
  print("Accuracy: " + str( truePredicted*100.0/N_test ) )

print('---------------------------- Naive Bayes ------------------------------------')
X_train, Y_train, X_test, Y_test, headers = processData(preprocessedData)
print("#### TRAINING DATA ####")
print(X_train)
print(Y_train)
print()
print("#### TESTING DATA ####")
print(X_test)
print(Y_test)
headers = headers[1:]
X_train = X_train[:, 1:]
X_test = X_test[:, 1:]
bayersian( X_train, Y_train, X_test, Y_test, headers, classAttributeIndex = classAttributeIndex )

