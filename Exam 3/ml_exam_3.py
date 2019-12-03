# -*- coding: utf-8 -*-
"""ML assignment 3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fxwiknetpWAy_2QvZuPo4NopMc4T7ssg

Data Cluster using K-means algorithm provided by the system.
1. Run k-means on AT&T 100 images, set K=10. Obtain confusion matrix. Re-order the confusion matrix using bipartite graph matching and obtain accuracy.
2. Run k-means on AT&T 400 images, set K=40. Obtain confusion matrix. Re-order the confusion matrix and obtain accuracy.
3. Run k-means on Hand-written-letters data, set K=26, as above.Computer 

Exam3 will depend on the codes you write for Project 3.

(A) Generate Three Gaussian distributions, each with 100 data points in 2 dimensions, with centers at (5,5), (-5, 5), and (-5,-5) and standard deviation \sigma = 2. Draw them in a Figure. Set K=3, do K-means clustering. Show the results in the same Figure. Repeat this 5 times. Submit the 5 figures, each represent the results of each K-means clustering. 

(B) Everything are same as (A), but with \sigma=4. Submit the 5 figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import munkres
import sys
import io

classes = [
  {
      "center": [ 5, 5 ],
      "variance": 4,
      "class" : 1,
      "size": 100
  },
  {
      "center": [ -5, 5 ],
      "variance": 4,
      "class" : 2,
      "size": 100
  },
  {
      "center": [ -5, -5 ],
      "variance": 4,
      "class" : 3,
      "size": 100
  }
]

m = munkres.Munkres()

# print(classes)

# Generate data of same class. Class is the last column
def generateData( size, mean, sigma, className ):
  Y = np.array( [ np.repeat( className, size ) ])
  for i in range( len(mean) ):
    new_X = np.array( [ np.random.normal( mean[i], sigma, size ) ] )
    if i == 0:
      X = new_X
    else:
      X = np.append( X, new_X , axis = 0 )
  X = np.append( X, Y, axis = 0 )
  return X.T

def generateMyDataSet( myObject, splitRatio = 0.2, plot = False ):

  # Generate data for each class
  for i in range( len(myObject) ):
    new_X = generateData( myObject[i]["size"], myObject[i]["center"], myObject[i]["variance"], myObject[i]["class"] )
    if i == 0:
      X = new_X
    else:
      X = np.append( X, new_X, axis = 0 )
    if plot == True:
      plt.subplot(3, 2, 1)
      plt.title('Initial Data')
      plt.scatter( new_X[:, 0], new_X[:, 1] )
  
  # Shuffling the data
  np.random.shuffle(X)
  N, M = np.shape( X )
  X, Y  = X[ : , :M-1 ], X[ : , M-1].reshape(-1,1)

  return X, Y

  # Split to train and test set
  N, M = np.shape( X )
  splitIndex = int(splitRatio * N)
  X_test,  Y_test  = X[ 0:splitIndex , :M-1 ], X[ 0:splitIndex , M-1]
  X_train, Y_train = X[ splitIndex: , :M-1 ], X[ splitIndex: , M-1]


  return X_train, Y_train, X_test, Y_test

# X_train, Y_train, X_test, Y_test = generateMyDataSet( classes )
X, Y = generateMyDataSet( classes )

def normalize( X ):
  meanX = np.mean( X, axis = 0 )
  stdX = np.std( X, axis = 0 )
  newX = ( X - meanX ) / stdX
  return newX

newX = normalize( X )

# Returns euclidean distance
def distance(pts, centroidPt):
  return ( np.sum( (pts - centroidPt)**2, axis = 1, keepdims = True ) ) ** 0.5

def getMyClass(Y):
  N, M = np.shape( Y )
  classes = dict()
  for i in range( N ):
    className = str( Y[i, 0] )
    if className not in classes:
      classes[ className ] = 0
    classes[ className ] += 1

  maxCount = 0
  maxClass = None
  for className in classes:
    if classes[ className ] > maxCount:
      maxCount = classes[ className ]
      maxClass = className
  
  return maxClass

def kmeans( X, Y, k = 3, plot = False, initCentroidsRandom = False ):
  
  change = 1
  threshold = 0.01
  X_N, X_M = np.shape( X )
  centroids = X[:k,:]
  if initCentroidsRandom:
    centroids = np.random.random( (k, X_M) )
  prev = np.ones( (X_N, 1) )

  iteration_index = 0
  while( True ):
    if change < threshold: break

    data = []
    p_centroids = copy.deepcopy(centroids)
    # Find distance with respect to all k centroids
    dist = distance( X, centroids[0, :] )
    for i in range( k-1 ):
      dist = np.append( dist, distance( X, centroids[i+1, :] ), axis = 1 )

    # Find the nearest k point
    nearestCentroid = np.argmin( dist, axis = 1 ).reshape(-1, 1)
    
    # Check the difference with previous set
    diff = prev == nearestCentroid
    prev = nearestCentroid
    myTrues = X_N - np.sum( diff )
    
    change = (myTrues / X_N)
    # print("Find points for each cluster:")
    for i in range( k ):
      myClusterPoints = X[ np.where( nearestCentroid == i )[0], : ]
      N, M = np.shape( myClusterPoints )
      if N > 0:
        centroids[i, :] = np.mean( myClusterPoints, axis = 0, keepdims=True )

        data.append({
            "centroid": p_centroids[i,:],
            "pts": myClusterPoints,
            "class": getMyClass( Y[ np.where( nearestCentroid == i )[0], : ] )
        })
      
    iteration_index += 1
    if plot == True and iteration_index < 6:
      plt.subplot(3, 2, iteration_index+1 )
      for c in data:
        plt.title('Iteration: '+ str(iteration_index) )
        plt.scatter( c["pts"][:, 0], c["pts"][:, 1] )
        plt.scatter( c["centroid"][0], c["centroid"][1], color = "black")

  return data

def accuracy( confusion_matrix ):
  return np.sum(np.trace(confusion_matrix))/np.sum(confusion_matrix)

def predict( X, Y, centroids_data ):

  N, M = np.shape( Y )
  classes = []
  for i in range( N ):
    if Y[i, 0] not in classes:
      classes.append( Y[i, 0] )
  
  confusion_matrix = np.zeros( ( len(classes), len(classes) ), dtype=int )

  for i in range( N ):
    actual_class = int( Y[i, 0] ) 

    minDistance = float( 'inf' )
    predicted_class = None
    for className in centroids_data:
      point = np.array( [ X[i,:] ] )
      centroid = np.array( [ className["centroid"] ] )
      dist = distance( point, centroid )[0][0]
      if( dist < minDistance ):
        minDistance = dist
        predicted_class = int( float( className["class"] ) )
    
    confusion_matrix[ actual_class - 1 ][ predicted_class - 1 ] += 1

  return confusion_matrix

def readFile( fileName ):
  with open( fileName, 'r' ) as f:
    lines = f.read().split( '\n' )
    return lines

# Converting the file data into a 2D array
def tabulateData( data, delimiter = ',', hasHeader = False ):
  X = []
  for line in data:
    words = line.split(delimiter)
    X.append(words)
  return X

# Removing data points which consists of null values
def preprocessData( tabulatedData ):
  X = []
  requiredLength = len( tabulatedData[0] )
  for dataPoint in tabulatedData:
    if( len(dataPoint) < requiredLength ):
      continue
    X.append( dataPoint[ :requiredLength ] )
  
  return X

# Function to split data into Training and Testing
def processData( preprocessedData ):

  # Converting to a numpy array
  preprocessedData = np.asarray( preprocessedData[:], dtype=int )
  Y = preprocessedData[ 0 , : ].reshape(-1, 1)
  X = preprocessedData[ 1: , : ]

  return X.T, Y


def generateCostMatrix( confusion_matrix ):
  return munkres.make_cost_matrix(confusion_matrix, lambda x : sys.maxsize - x)

def bipartile( cost_matrix ):
  indexes = m.compute(cost_matrix)
  return indexes

# 3A. PartA
datasetFileName = "Dataset/HandWrittenLetters.txt"
k = 26
print("************** 3A. PartA ****************")
fileData = readFile( datasetFileName )
tabulatedData = tabulateData( fileData )
preprocessedData = preprocessData( tabulatedData )

X, Y = processData( preprocessedData )

centroids_data = kmeans( X, Y, k, plot = False, initCentroidsRandom = True )
confusion_matrix = predict( X, Y, centroids_data )
print( "Accuracy before: " + str ( accuracy(confusion_matrix) ) )
print( "Confusion Matrix before: ")
print(confusion_matrix)
cost_matrix = generateCostMatrix( confusion_matrix )
indexes = bipartile( cost_matrix )

for x in indexes:
  actual  = str( int( x[1] + 1 ) )
  replace = str( int( x[0] + 1 ) )
  if(actual == replace):
    continue
  for y in centroids_data:
    if( y["class"] == actual ):
      y["class"] = replace
      break
print("---------------------------------------------------------")
confusion_matrix2 = predict( X, Y, centroids_data )
print( "Accuracy after: " + str ( accuracy(confusion_matrix2) ) )
print( "Confusion Matrix after: ")
print(confusion_matrix2)


def findObjectiveFunctionValue( centroids_data ):
  output = 0
  for c in centroids_data:
    output += np.sum( distance( c["pts"],  c["centroid"] ) )
  return output

# 3A. PartB

datasetFileName = "Dataset/HandWrittenLetters.txt"
k = 26
fileData = readFile( datasetFileName )
tabulatedData = tabulateData( fileData )
preprocessedData = preprocessData( tabulatedData )

X, Y = processData( preprocessedData )

minObjectiveValue = float("inf")
centroids_data = None
print("************** 3A. PartB ****************")
print("Objective Values: ")
for z in range( 10 ):
  centroids_data_k = kmeans( X, Y, k, plot = False, initCentroidsRandom = True )
  myObjectiveValue = findObjectiveFunctionValue( centroids_data_k )
  print("Iteration "+ str(z) + " : " + str(myObjectiveValue))
  if myObjectiveValue < minObjectiveValue:
    minObjectiveValue = myObjectiveValue
    centroids_data = centroids_data_k

print("-------------")
print("Selected objective value: " + str(minObjectiveValue))


confusion_matrix = predict( X, Y, centroids_data )
print( "Accuracy before: " + str ( accuracy(confusion_matrix) ) )
print( "Confusion Matrix before: ")
print(confusion_matrix)
cost_matrix = generateCostMatrix( confusion_matrix )
indexes = bipartile( cost_matrix )

for x in indexes:
  actual  = str( int( x[1] + 1 ) )
  replace = str( int( x[0] + 1 ) )
  if(actual == replace):
    continue
  for y in centroids_data:
    if( y["class"] == actual ):
      y["class"] = replace
      break
print("-----")
confusion_matrix2 = predict( X, Y, centroids_data )
print( "Accuracy after: " + str ( accuracy(confusion_matrix2) ) )
print( "Confusion Matrix after: ")
print(confusion_matrix2)