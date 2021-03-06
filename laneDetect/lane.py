import cv2
import numpy as np
from utils import Util
import pyttsx3 as p


engine = p.init()


class Lane:
    def __init__(self, path):
        self.path = path
        self.util = Util()
    
    def run_img(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (800, 600))
        #self.detect(img)
        cv2.imshow('Frame', img)


    def run(self, path):
        cap = cv2.VideoCapture(path)
        out = cv2.VideoWriter('output.mp4',0x7634706d , 20.0, (640,480))
    
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                # Display the resulting frame
                frame = cv2.resize(frame, (800, 600))
                out.write(frame)
                self.detect(frame)
                cv2.imshow('Frame', frame)
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def detect(self, screen):
        vert = np.array(
            [[100, 550], [375, 350], [450, 350], [800, 550]], np.int32)
        fin = self.util.edgeDetect(screen)
        fin = self.util.roi(fin, [vert])
        line = cv2.HoughLinesP(fin, 2, np.pi/180, 20, 7, 7)
        if not(line is None):
            for i in line:
                cv2.line(screen, (i[0][0], i[0][1]),
                         (i[0][2], i[0][3]), (255, 0, 0), 10)
        l1dataset = []
        l2dataset = []
        try:
            straightxcors, straightycors = self.util.averageLanes(line)
            xcors, ycors = self.util.getPoints(line)

            l1dataset.append(straightxcors[0])
            l1dataset.append(straightycors[0])
            l2dataset.append(straightxcors[1])
            l2dataset.append(straightxcors[1])
            allstraightxcors = straightxcors[0] + straightxcors[1]
            allstraightycors = straightycors[0] + straightycors[1]

            l1m, l1b = self.util.linearRegression(l1dataset[0], l1dataset[1])
            l2m, l2b = self.util.linearRegression(l2dataset[0], l2dataset[1])

            allm, allb = self.util.linearRegression(
                allstraightxcors, allstraightycors)
            allxcor1 = int((allm * 350) + allb)
            allxcor2 = int(allb)

            filterl1x = []
            filterl1y = []
            filterl2x = []
            filterl2y = []

            for count, i in enumerate(ycors):
                if (i*l2m + l2b < xcors[count]):
                    filterl2x.append(xcors[count])
                    filterl2y.append(i)
                else:
                    filterl1x.append(xcors[count])
                    filterl1y.append(i)

            l1inx1 = int((600 - l1b) / l1m)
            l1inx2 = int((350-l1b) / l1m)

            l2inx1 = int((600-l2b) / l2m)
            l2inx2 = int((350-l2b) / l2m)

            cv2.line(screen, (int(l1inx1), 600),
                     (int(l1inx2), 350), (0, 0, 0), 10)
            cv2.line(screen, (int(l2inx1), 600),
                     (int(l2inx2), 350), (0, 0, 0), 10)

            cv2.line(screen, (allxcor1, 600), (allxcor2,350), (255,0,0), 10)
            turning = ""

            results = self.util.intersection([l1m, l1b], [l2m, l2b])
            if not (results is None):
                if (results[0] > 400):
                    with open("write.txt", "w") as f:
                        f.write("Turn Left")
                else:
                    with open("write.txt", "w") as f:
                        f.write("Turn Right")
            else:
                with open("write.txt", "w") as f:
                    f.write("Go straight")
            try:
                equ1, polyx1, polyy1 = self.util.polyReg(filterl2x, filterl2y)

                for i in range(len(polyx1)):
                    if i == 0:
                        pass
                    else:
                        cv2.line(screen, (int(polyx1[i]), int(polyy1[i])), (int(
                            polyx1[i-1]), int(polyy1[i-1])), (255, 255, 0), 10)
            except Exception as e:
                print(e)
            try:
                equ2, polyx2, polyy2 = self.util.polyReg(filterl1x, filterl1y)

                for i in range(len(polyx2)):
                    if i == 0:
                        pass
                    else:
                        cv2.line(screen, (int(polyx2[i]), int(polyy2[i])), (int(
                            polyx2[i-1]), int(polyy2[i-1])), (255, 255, 0), 10)

            except:
                pass

        except Exception as e:
            pass

        return screen
