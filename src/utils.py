import cv2
import numpy as np
import scipy


class Util:
    def __init__(self, video):
        pass

    # tells if the slope is increasing or decreasing.
    def navigationSlope(self, x1, y1, x2, y2):
        if not x1 == x2:
            a = ((y2-y1)/(x2-x1)) >= 2  # positive slope
            return a

    def lineSlope(self, x1, y1, x2, y2):  # gives the slope of a line
        m = (y2 - y2) / (x2 - x1)
        return m

    def intersection(self, line1, line2):  # finds point of intersection of two lines
        m = self.lineSlope(line1[0], line1[1], line2[0], line2[1])
        if (m <= 800 and m >= 0) and (((line1[0]*m) + line1[1]) >= 0) and ((line1[0]*m) + line1[1] <= 600):
            return [m, (line1[0]*m) + line1[1]]

    def getPoints(lines):  # gets points on the line
        xcors = []  # x co-ordinate
        ycors = []  # y co-ordinate
        for i in lines:
            xcors.append(i[0][0])
            ycors.append(i[0][1])
            xcors.append(i[0][2])
            ycors.append(i[0][3])
        return xcors, ycors

    def linearRegression(self, X, Y):
        mean_x = np.mean(X)
        mean_y = np.mean(Y)
        m = len(X)
        numer = 0
        denom = 0
        for i in range(m):
            numer += (X[i] - mean_x) * (Y[i] - mean_y)
            denom += (X[i] - mean_x) ** 2
        b1 = numer / denom
        b0 = mean_y - (b1 * mean_x)

        max_x = np.max(X) + 5
        min_x = np.min(X) - 5
        x = np.linspace(min_x, max_x, 1000)
        y = b0 + b1 * x

        return b1, b0

    def polyReg(xcors, ycors):
        def func(x, a, b, c): return a*(x**2) + (b*x) + c
        time = np.array(xcors)
        avg = np.array(ycors)
        initialGuess = [5, 5, -.01]  # random
        popt, pcov = scipy.optimize.curve_fit(func, time, avg, initialGuess)
        cont = np.linspace(min(time), max(time), 50)
        fittedData = [func(x, *popt) for x in cont]
        xcors = []
        ycors = []
        for count, i in enumerate(cont):
            xcors.append(i)
            ycors.append(fittedData[count])
        return popt, xcors, ycors

    def average(diction):
        xcors1, ycors1, xcors2, ycors2, count = 0
        for data in diction:
            xcors1 = xcors1 + data[2][0]
            ycors1 = ycors1 + data[2][1]
            xcors2 = xcors2 + data[2][2]
            ycors2 = ycors2 + data[2][3]
            count = count + 1
        xcors1 = xcors1/count
        ycors1 = ycors1/count
        xcors2 = xcors2/count
        ycors2 = ycors2/count
        return (int(xcors1), int(ycors1), int(xcors2), int(ycors2))
