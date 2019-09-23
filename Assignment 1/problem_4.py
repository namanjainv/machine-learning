import numpy as np
import matplotlib.pyplot as plt

# Constants
N = 10
sin_curve_points = 100
seed = 1000
M = 5

np.random.seed(seed)

def generatePoints(N, degree):
    # Generating X values in the range(0,1) with a step of 1/N
    X = np.ones(shape=np.shape(pts), dtype=float)
    for i in range(degree):
        X = np.append(X, pts**(i+1), axis=1)

    # Generating noise - StandX_testard Gaussian Distribution
    noise = np.random.normal(0,0.1, size=(N,1))

    # Find Y = sin(2*pi*x) + noise
    Y = np.sin(2*np.pi*pts) + noise

    return X, Y

def plotGraph(X, Y, title, subplot_index, w = None, degree = None):
    x = np.arange(0, 1, 1.0/sin_curve_points, dtype=float).reshape(-1,1)
    sin_curve_y = np.sin(2*np.pi*x).reshape(-1,1)
    plt.subplot(1, 2, subplot_index)
    if w is not None:
        X_test = np.ones(shape=np.shape(x), dtype=float)
        for i in range(degree):
            X_test = np.append(X_test, x**(i+1), axis=1)
        fit_curve_y = predict(X_test, w)
        plt.plot(x, fit_curve_y, color='red')
    plt.scatter(X, Y, color='blue')
    plt.plot(x, sin_curve_y, color='green')

def init_weights(degree):
    w = np.ones(shape=(1, degree+1), dtype=float)*10000
    return w

def compute_t(X, w):
    return sum(np.dot(w, np.transpose(X))).reshape(-1,1)

def fit(X, Y, w, lr = 0.1, threshold = 0.0001):
    pE = 0
    while(True):
        t = compute_t(X, w)
        E = 0.5*sum((Y - t)**2)/N
        if E[0] <= threshold:
            break
        if np.abs(E[0] - pE) < threshold**2:
            break
        pE = E[0]
        dw = np.dot(np.transpose(Y - t), X)
        w = w + (lr*dw)
    return w

def predict(X, w):
    return compute_t(X, w)

# Initialization
pts = np.arange(0,1,1.0/N,dtype=float).reshape(-1,1)
X, Y = generatePoints(N, M)
w = init_weights(M)
w = fit(X, Y, w)
print(w)
plotGraph(pts, Y, 'Actual', 1)
plotGraph(pts, Y, 'M = ' + str(M), 2, w, M)
plt.show()