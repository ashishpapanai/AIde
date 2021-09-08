import os
import re
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds


class Utils:
    def __init__(self):
        pass

    def swap_xy(self, boxes):
        return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)

    def convert_to_xywh(self, boxes):
        return tf.concat(
            [(boxes[..., :2] + boxes[..., 2:]) /
             2.0, boxes[..., 2:] - boxes[..., :2]],
            axis=-1,
        )

    def convert_to_corners(self, boxes):
        return tf.concat(
            [boxes[..., :2] - boxes[..., 2:] / 2.0,
                boxes[..., :2] + boxes[..., 2:] / 2.0],
            axis=-1,
        )
