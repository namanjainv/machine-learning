# Search for K nearest Points
# Write dist(x_i, x_j) where x_i, x_j are data instances
# p as dimensions of data
# Write dist(x_i_k, x_j_k) where k is the attribute 
# Note: Last element is a class

# Add a function to normalize too
import numpy as np
import re
import sys

# Categorical Values - Eg: (Rainy, Sunny, Cloudy) - Same implies 0 else 1 - Hamming Distance
# Ordinal - Eg: Text or [A, B, C, D] - Manhattan Distance, A to B = 1, but A to C is 2
# Numerical - Euclidean Distance

# Procedure:
# 1. Read the complete training file and see each attributes
# 2. For each attributes categorize between Ordinal, Numerical and Categorical and store them in list
# 3. For each point in test file, compare with every data point in training set and find the least k neighbors
# 4. Check the class of all the k neighbors and assign the best class to it

# def distance_ordinal(ptA_attribute_value, ptB_attribute_value):
#     return abs(ord(ptA_attribute_value) - ord(ptB_attribute_value))

def distance_ordinal(ptA_ordinal_values, ptB_ordinal_values):
    return np.sum(np.abs(ptA_ordinal_values - ptB_ordinal_values))

def distance_categorical(ptA_categorical_values, ptB_categorical_values):
    diff = ptA_categorical_values == ptB_categorical_values
    return np.size(diff) - np.sum(diff)
    
def distance_euclidean(ptA_numerical_values, ptB_numerical_values):
    a = (ptA_numerical_values)
    b = (ptB_numerical_values)
    return (np.sum((a - b)**2)**0.5)

# Constants
ordinal_attributes = []
categorical_attributes = []
numeric_attributes = []
class_attribute = None

def findAttributeTypes(array):
    for i in range(len(array)):
        regex_output = None
        x = re.search('^[A-Za-z]{1}$', array[i])
        if x is not None:
            ordinal_attributes.append(i)
            continue
        x = re.search('^[A-Za-z]+[-]*', array[i])
        if x is not None:
            categorical_attributes.append(i)
            continue
        else:
            numeric_attributes.append(i)

def distance(ptA, ptB):
    ptA_numerical_values   = np.array([ float(ptA[attr]) for attr in numeric_attributes ])
    ptA_oridinal_values    = np.array([ ord(ptA[attr]) for attr in ordinal_attributes ])
    ptA_categorical_values = np.array([ ptA[attr] for attr in categorical_attributes ])

    ptB_numerical_values   = np.array([ float(ptB[attr]) for attr in numeric_attributes ])
    ptB_oridinal_values    = np.array([ ord(ptB[attr]) for attr in ordinal_attributes ])
    ptB_categorical_values = np.array([ ptB[attr] for attr in categorical_attributes ])

    dist = 0
    dist += (distance_euclidean(ptA_numerical_values, ptB_numerical_values))
    dist += (distance_categorical(ptA_categorical_values, ptB_categorical_values))
    dist += (distance_ordinal(ptA_oridinal_values, ptB_oridinal_values))

    return dist

def prepareData(filename, class_attribute, getClassCount = False):
    n_classes = []
    X = []
    Y = []
    f = open(filename, "r")
    i = 0
    for line in f:
        array = line.strip().split(" ")
        if not array[class_attribute] in n_classes:
            n_classes.append(array[class_attribute])
        Y.append(array[class_attribute])
        del array[class_attribute]
        X.append(array)
    if(getClassCount):
        return X, Y, n_classes
    return X, Y

def knn(testPoint, X_train, Y_train, k = 1):

    def getMaxDistance(array): 
        maxObj = None
        max_dist = -1
        max_index = -1
        i = 0
        for obj in array:
            if( obj.get('distance') > max_dist ):
                max_dist = obj.get('distance')
                max_index = i
                maxObj = obj
            i += 1
        return max_index
    

    k_nearest_nodes = []
    min_dist = 9999999
    min_index = -1
    for i in range(np.shape(Y_train)[0]):
        trainPoint = X_train[i]
        dist = distance(trainPoint, testPoint)
        if(min_dist > dist):
            if(len(k_nearest_nodes) == k):
                del k_nearest_nodes[max_obj_index]
            k_nearest_nodes.append( { "index": i, "distance": dist, "point": X_train[i], "class": Y_train[i] } )
            if(len(k_nearest_nodes) == k):
                max_obj_index = getMaxDistance(k_nearest_nodes)
                obj = k_nearest_nodes[max_obj_index]
                min_dist = obj.get('distance')
                
    # print( str(testPoint) + " has following " + str(k) + " near points: " )
    # print( k_nearest_nodes )
    # print("---------------------------------------------------------------")
    return k_nearest_nodes

def initializeHashMap(classes):
    classMap = {}
    for class_name in classes:
        classMap[class_name] = 0 
    return classMap

def __main__(argv):
    # try:
    train_file = argv[1]
    test_file = argv[2]
    class_attribute = int(argv[3])
    X_train, Y_train, n_classes = prepareData(train_file, class_attribute, True)
    X_test, Y_test = prepareData(test_file, class_attribute)

    findAttributeTypes(X_train[0])
    print("Oridnal Attributes: " + str(ordinal_attributes))
    print("Categorical Attributes: " + str(categorical_attributes))
    print("Numeric Attributes: " + str(numeric_attributes))
    print("Classes: " + str(n_classes) )

    for i in range(np.shape(Y_test)[0]):
        myMap = initializeHashMap(n_classes)
        testPoint = X_test[i]
        print(testPoint)
        print(myMap)
        k_nearest_nodes = knn(testPoint, X_train, Y_train, k = 2)
        for near_node in k_nearest_nodes:
            class_name = near_node.get("class")
            myMap[class_name] += 1
        print(myMap)
    # except:
        # print("Run the program with: python3 main.py Dataset/train.txt Dataset/test.txt 2")


# ------------------------ TESTING -------------------------
# test_a = "Rain E 1 1"
# test_b = "Rainy C 0 0"
# test_a_array = test_a.split(" ")
# test_b_array = test_b.split(" ")
# findAttributeTypes(test_a_array)
# # print("Ordinal Attributes: "+str(ordinal_attributes))
# # print("Numeric Attributes: "+str(numeric_attributes))
# # print("Categorical Attributes: "+str(categorical_attributes))

# d = distance(test_a_array, test_b_array)
# print(d)

__main__(sys.argv)
