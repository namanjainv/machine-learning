import numpy as np
import matplotlib.pyplot as plt
import math

# Constants
N = 10
sin_curve_points = 100
seed = 0
m = [9]

# Generating random seed
np.random.seed(seed)

# Initialization
pts = np.linspace(0,1,N).reshape(-1,1)

# Generating noise - StandX_testard Gaussian Distribution
power = np.arange(N).reshape(-1,1)
noise = (0.2+np.random.random(N).reshape(-1,1))*(np.ones(N).reshape(-1,1)*-1)**power

# Find Y = sin(2*pi*x) + noise
Y = np.sin(2*np.pi*pts) + noise

# Function to generate N points
# param degree - Attributes with degree 0 ... degree
def generatePoints(degree):
  
    # Generating X values in the range(0,1) with a step of 1/N
    X = np.ones(shape=np.shape(pts), dtype=float)
    
    for i in range(degree):
        X = np.append(X, pts**(i+1), axis=1)

    return X, Y

print("Points:")
print(pts)
print("------")

print("\nSin Value with noise:")
print(Y)
print("------")

# Generating sine curve points
x = np.arange(0, 1, 1.0/sin_curve_points, dtype=float).reshape(-1,1)
sin_curve_y = np.sin(2*np.pi*x).reshape(-1,1)

def plotGraph(X, Y, title, subplot_index, w = None, degree = None):
    delta = 0.1
    legend_handles = []
    plt.subplot(3, 3, subplot_index)

    # Setting the dimensions
    plt.ylim( -4-delta, 4+delta)
    plt.xlim( -delta, 1+delta)
    plt.title(title)

    # Plotting the points
    plt.scatter(X, Y, label="Data Points", color='blue')
    
    # Plotting the sine curve
    plt.plot(x, sin_curve_y, label="Sine Curve", color='green')
    
    
    # Plotting the fit line
    if w is not None:
        X_test = np.ones(shape=np.shape(x), dtype=float)
        for i in range(degree):
            X_test = np.append(X_test, x**(i+1), axis=1)
        fit_curve_y = predict(X_test, w)
        plt.plot(x, fit_curve_y, label="Fit Curve", color='red')
        
    
    plt.legend(loc='upper right')

def predict(X, w):
    return np.dot(X, w).reshape(-1,1)

"""# Without Regularization"""

print("Without Regularization")
plt.rcParams['figure.figsize'] = [18.0, 12.0]

for i in range(len(m)):
    M = m[i]
    X, Y = generatePoints(M)
    n_col = X.shape[1]
    w = np.linalg.lstsq(X, Y)[0]
    print("Degree: "+str(M))
    print("Weights:")
    print(w)
    print("------------------")
    plotGraph(pts, Y, 'Degree: '+str(M), i+2, w = w, degree = M)
plotGraph(pts, Y, 'Actual', 1)
plt.show()

"""# With Regularization"""

# Ref https://en.wikipedia.org/wiki/Regularized_least_squares
print("With Regularization")
_lambda_values = [10**-20, 10**-10, 10**-6, 10**-4, 10**-2, 0] 
M = 9
plt.rcParams['figure.figsize'] = [18.0, 12.0]
for i in range(len(_lambda_values)):
    _lambda = _lambda_values[i]
    X, Y = generatePoints(M)
    n_col = X.shape[1]
    w = np.linalg.lstsq(X.T.dot(X) + _lambda * np.identity(n_col), X.T.dot(Y))[0]
    print("Degree: "+str(M))
    print("Lambda: "+str(_lambda))
    print("Weights:")
    print(w)
    print("------------------")
    plotGraph(pts, Y, 'Degree: '+str(M)+', Lambda: '+ str(_lambda), i+2, w = w, degree = M)
plotGraph(pts, Y, 'Actual', 1)
plt.show()

