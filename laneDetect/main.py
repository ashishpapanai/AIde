from numpy.lib import utils
from lane import Lane
import cv2
import time
import pyttsx3 as p
from utils import Util

if __name__ == "__main__":
    video = cv2.VideoCapture(r'\data\data.mp4') # replace with the full path of video
    lane = Lane(video)
    lane.run(video)
    util = Util()
    util.speak()

    