from numpy.lib import utils
from lane import Lane
import cv2
import time
import pyttsx3 as p
from utils import Util

if __name__ == "__main__":
    video_path = r'E:\road-vision-ai\data\test_road.mp4' # replace with the full path of video
    lane = Lane(video_path)
    lane.run(path=video_path)
    '''image_path = r'E:\road-vision-ai\data\test123.jpg'
    lane = Lane(image_path)
    lane.run(path=image_path)'''
    util = Util()
    util.speak()

        