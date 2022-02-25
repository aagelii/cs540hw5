import csv
import numpy as np
from matplotlib import pyplot as plt
import math


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    """
    INPUT: 
        filename - a string representing the path to the csv file.

    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    dataset = []
    with open(filename, newline='') as csvfile:
        next(csvfile)
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append(row[1:])
    dataset = np.array(dataset)
    return dataset


def print_stats(dataset, col):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on. 
                  For example, 1 refers to density.

    RETURNS:
        None
    """
    sum = 0
    count = 0
    for i in dataset:
        sum += float(i[col])
        count += 1
    mean = sum / count
    sdSum = 0
    for j in dataset:
        sdSum += pow(float(j[col]) - mean, 2)
    print(count)
    print(f'{mean:.2f}')
    print(f'{math.sqrt((1.0 / (count - 1)) * sdSum):.2f}')
    pass


def regression(dataset, cols, betas):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        mse of the regression model
    """
    mse = 0
    for i in dataset:
        temp = betas[0]
        count = 1
        for j in cols:
            temp += float(i[j]) * betas[count]
            count += 1
        mse += pow(temp - float(i[0]), 2)
    mse *= (1.0 / len(dataset))
    return mse


def gradient_descent(dataset, cols, betas):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        An 1D array of gradients
    """
    grads = []
    for k in range(len(betas)):
        mse = 0
        for i in dataset:
            temp = betas[0]
            count = 1
            for j in cols:
                temp += float(i[j]) * betas[count]
                count += 1
            if k == 0:
                mse += (temp - float(i[0]))
            else:
                mse += (temp - float(i[0])) * float(i[cols[k - 1]])
        mse *= (2.0 / len(dataset))
        grads.append(mse)
    grads = np.array(grads)
    return grads


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """

    for i in range(0, T):
        newGrad = []
        grad = gradient_descent(dataset, cols, betas)
        for j in range(len(betas)):
            newGrad.append(betas[j] - eta * (grad[j]))
        betas = newGrad
        printStr = str(i + 1) + " " + str(f'{regression(dataset, cols, betas):.2f}')
        for k in newGrad:
            printStr += " " + str(f'{k:.2f}')
        print(printStr)
    pass


def compute_betas(dataset, cols):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    betas = []
    rows = []
    y = []
    for i in dataset:
        temp = [1]
        y.append(float(i[0]))
        for j in cols:
            temp.append(float(i[j]))
        rows.append(temp)
    X = np.array(rows)
    y = np.array(y)
    Xt = np.transpose(X)
    tempMat = np.matmul(Xt, X)
    tempMat = np.linalg.inv(tempMat)
    tempMat = np.matmul(tempMat, Xt)
    tempMat = np.matmul(tempMat, y)
    mse = regression(dataset, cols, tempMat)
    for i in tempMat:
        betas.append(float(i))
    return mse, *betas


def predict(dataset, cols, features):
    """
    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    betas = compute_betas(dataset, cols)
    result = betas[1]
    count = 2
    for i in features:
        result += i * betas[count]
        count += 1
    return result


def synthetic_datasets(betas, alphas, X, sigma):
    """
    Input:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (n,1))
        sigma  - standard deviation of noise

    RETURNS:
        Two datasets of shape (n,2) - linear one first, followed by quadratic.
    """

    linear = []
    quadratic = []
    for x in X:
        zeta = np.random.normal(0, sigma)
        linear.append([betas[0] + betas[1] * x[0] + zeta, x[0]])
        quadratic.append([alphas[0] + alphas[1] * pow(x[0], 2) + zeta, x[0]])
    linear = np.array(linear)
    quadratic = np.array(quadratic)
    return linear, quadratic


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')
    figure = plt.figure()
    X = []
    for i in range(1000):
        X.append([np.random.randint(-100, 101)])
    X = np.array(X)
    alphas = np.array([-1, 1])
    betas = np.array([1, 2])
    sigmas = np.array([0.0001, .001, .01, .1, 1, 10, 100, 1000, 10000, 100000])
    plt.xlabel("Standard Deviation of Error Term")
    plt.ylabel("MSE of Trained Model")
    mseLinA = []
    mseQuadA = []
    for sig in sigmas:
        datasets = synthetic_datasets(betas, alphas, X, sig)
        mseLin = compute_betas(datasets[0], cols=[1])
        mseQuad = compute_betas(datasets[1], cols=[1])
        mseLinA.append(mseLin[0])
        mseQuadA.append(mseQuad[0])
    plt.plot(sigmas, mseLinA, 'o-', label="MSE of Linear Dataset")
    plt.plot(sigmas, mseQuadA, 'o-', label="MSE of Quadratic Dataset")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig("mse.pdf")
    plt.show()


if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()
