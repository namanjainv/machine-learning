import numpy as np
import matplotlib.pyplot as plt

# Constants
N = 10
sin_curve_points = 100
seed = 0
m = [0,2,4,6,9]

np.random.seed(seed)
# Generating noise - StandX_testard Gaussian Distribution
noise = np.random.normal(0,0.05, size=(N,1))

# Generating sine curve points
x = np.arange(0, 1, 1.0/sin_curve_points, dtype=float).reshape(-1,1)
sin_curve_y = np.sin(2*np.pi*x).reshape(-1,1)

def generatePoints(N, degree):
    # Generating X values in the range(0,1) with a step of 1/N
    X = np.ones(shape=np.shape(pts), dtype=float)
    for i in range(degree):
        X = np.append(X, pts**(i+1), axis=1)

    # Find Y = sin(2*pi*x) + noise
    Y = np.sin(2*np.pi*pts) + noise

    return X, Y

def plotGraph(X, Y, title, subplot_index, w = None, degree = None):
    delta = 0.2
    plt.subplot(2, 3, subplot_index)

    # Setting the dimensions
    plt.ylim( -1-delta, 1+delta)
    plt.xlim( -delta, 1+delta)
    plt.title('Actual')

    # Plotting the fit line
    if w is not None:
        X_test = np.ones(shape=np.shape(x), dtype=float)
        for i in range(degree):
            X_test = np.append(X_test, x**(i+1), axis=1)
        fit_curve_y = predict(X_test, w)
        plt.plot(x, fit_curve_y, color='red')

        # Settings
        plt.title('Degree: '+str(degree))
    
    # Plotting the points
    plt.scatter(X, Y, color='blue')

    # Plotting the sine curve
    plt.plot(x, sin_curve_y, color='green')


def compute_t(X, w):
    return np.dot(X, w).reshape(-1,1)

def predict(X, w):
    return compute_t(X, w)

# Initialization
pts = np.arange(0,1,1.0/N,dtype=float).reshape(-1,1)
for i in range(len(m)):
    M = m[i]
    X, Y = generatePoints(N, M)
    w = np.linalg.lstsq(X, Y)[0]
    plotGraph(pts, Y, '', i+2, w = w, degree = M)
    print(M)
    print(w)
    print("--------")
plotGraph(pts, Y, '', 1)

plt.show()