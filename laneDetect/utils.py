import cv2
import numpy as np
from scipy.optimize import curve_fit
import pyttsx3 as p
import time
from numpy.linalg import lstsq


class Util:
    def __init__(self):
        self.flag = 0 

    # tells if the slope is increasing or decreasing.
    def navigationSlope(self, x1, y1, x2, y2):
        if not(x1 == x2):
            a = not (((y2-y1)/(x2-x1)) < 0)
        return a

    def lineSlope(self, x1, y1, x2, y2):  # gives the slope of a line
        m = (y2-y1) / (x2-x2)
        return m

    def intersection(self, line1, line2):  # finds point of intersection of two lines
        x = (line2[1]-line1[1])/(line1[0]-line2[0])
        if(x <= 800 and x >= 0) and (((line1[0]*x) + line1[1]) >= 0) and ((line1[0]*x) + line1[1] <= 600):
            return [x, (line1[0]*x) + line1[1]]

    def getPoints(self, lines):  # gets points on the line
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

    def polyReg(self, xcors, ycors):
        def func(x, a, b, c): return a*(x**2) + (b*x) + c
        time = np.array(xcors)
        avg = np.array(ycors)
        initialGuess = [5, 5, -.01]  # random
        popt, pcov = curve_fit(func, time, avg, initialGuess)
        cont = np.linspace(min(time), max(time), 50)
        fittedData = [func(x, *popt) for x in cont]
        xcors = []
        ycors = []
        for count, i in enumerate(cont):
            xcors.append(i)
            ycors.append(fittedData[count])
        return popt, xcors, ycors

    def average(self, diction):
        xcors1 = 0
        ycors1 = 0
        xcors2 = 0
        ycors2 = 0
        count = 0
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

    def roi(self, img, vert):  # region of interest
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vert, 255)
        return cv2.bitwise_and(img, mask)

    def edgeDetect(self, img):  # detect edges in the image
        edges = cv2.Canny(img, 250, 300)
        #print(cv2.GaussianBlur(edges, (3,3), 0))
        return cv2.GaussianBlur(edges, (3, 3), 0)

    def averageLanes(self, lines):
        ycor = []

        for i in lines:
            for x in i:
                ycor.append(x[1])
                ycor.append(x[3])

        minY = min(ycor)
        maxY = 600
        linesDict = {}
        finalLines = {}
        lineCount = {}
        for count, i in enumerate(lines):
            for x in i:

                xcors = (x[0], x[2])
                ycors = (x[1], x[3])

                A = np.vstack([xcors, np.ones(len(xcors))]).T
                m, b = lstsq(A, ycors, rcond=None)[0]

                x1 = (minY-b) / m
                x2 = (maxY-b) / m

                linesDict[count] = [m, b, [int(x1), minY, int(x2), maxY]]

        status = False
        for i in linesDict:
            finalLinesCopy = finalLines.copy()

            m = linesDict[i][0]
            b = linesDict[i][1]

            line = linesDict[i][2]

            if len(finalLines) == 0:
                finalLines[m] = [[m, b, line]]
            else:
                status = False

                for x in finalLinesCopy:
                    if not status:
                        if abs(x*1.2) > abs(m) > abs(x*0.8):
                            if abs(finalLinesCopy[x][0][1]*1.2) > abs(b) > abs(finalLinesCopy[x][0][1]*0.8):
                                finalLines[x].append([m, b, line])
                                status = True
                                break

                        else:
                            finalLines[m] = [[m, b, line]]

        for i in finalLines:
            lineCount[i] = len(finalLines[i])

        extremes = sorted(lineCount.items(), key=lambda item: item[1])[
            ::-1][:2]
        lane1 = extremes[0][0]
        lane2 = extremes[1][0]

        l1x1, l1y1, l1x2, l1y2 = self.average(finalLines[lane1])
        l2x1, l2y1, l2x2, l2y2 = self.average(finalLines[lane2])

        allxcors = [[l1x1, l1x2], [l2x1, l2x2]]
        allycors = [[l1y1, l1y2], [l2y1, l2y2]]

        return allxcors, allycors

    def speak(self, path = "./write.txt"):
        engine = p.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id) 
        while True:
            with open(path, "r") as f:
                s = f.read()
            if s == "Done":
                break
            if self.flag < 3:
                engine.say(s)
                engine.runAndWait()
                time.sleep(1)
                self.flag = self.flag + 1
            else:
                engine.stop()
                break
