import cv2
import numpy as np
import math
import scipy
from utils import Util


class Lane:
    def __init__(self, video):
        self.data = video
        self.util = Util(video)
        
    

    