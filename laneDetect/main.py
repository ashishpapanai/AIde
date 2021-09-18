from numpy.lib import utils
from lane import Lane
import cv2
import time
import pyttsx3 as p
from utils import Util

if __name__ == "__main__":
    video_path = r'E:\road-vision-ai\data\data.mp4' # replace with the full path of video
    lane = Lane(video_path)
    lane.run(path=video_path)
    util = Util()
    util.speak()

        