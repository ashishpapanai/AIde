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

    def compute_iou(self, boxes1, boxes2):
        boxes1_corners = self.convert_to_corners(boxes1)
        boxes2_corners = self.convert_to_corners(boxes2)
        lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
        rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
        intersection = tf.maximum(0.0, rd - lu)
        intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
        boxes1_area = boxes1[:, 2] * boxes1[:, 3]
        boxes2_area = boxes2[:, 2] * boxes2[:, 3]
        union_area = tf.maximum(
            boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
        )
        return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)

    def display(
        image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[0, 0, 1]
    ):
        image = np.array(image, dtype=np.uint8)
        plt.figure(figsize=figsize)
        plt.axis("off")
        plt.imshow(image)
        ax = plt.gca()
        for box, _cls, score in zip(boxes, classes, scores):
            text = "{}: {:.2f}".format(_cls, score)
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            patch = plt.Rectangle(
                [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
            )
            ax.add_patch(patch)
            ax.text(
                x1,
                y1,
                text,
                bbox={"facecolor": color, "alpha": 0.4},
                clip_box=ax.clipbox,
                clip_on=True,
            )
        plt.show()
        return ax

    def random_flip_horizontal(self, image, boxes):
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            boxes = tf.stack(
                [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
            )
        return image, boxes


    def resize_and_pad_image(
        self, image, min_side=800.0, max_side=1333.0, jitter=[640, 1024], stride=128.0
    ):
        image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
        if jitter is not None:
            min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
        ratio = min_side / tf.reduce_min(image_shape)
        if ratio * tf.reduce_max(image_shape) > max_side:
            ratio = max_side / tf.reduce_max(image_shape)
        image_shape = ratio * image_shape
        image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
        padded_image_shape = tf.cast(
            tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
        )
        image = tf.image.pad_to_bounding_box(
            image, 0, 0, padded_image_shape[0], padded_image_shape[1]
        )
        return image, image_shape, ratio


    def preprocess_data(self, sample):
        image = sample["image"]
        bbox = self.swap_xy(sample["objects"]["bbox"])
        class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)

        image, bbox = self.random_flip_horizontal(image, bbox)
        image, image_shape, _ = self.resize_and_pad_image(image)

        bbox = tf.stack(
            [
                bbox[:, 0] * image_shape[1],
                bbox[:, 1] * image_shape[0],
                bbox[:, 2] * image_shape[1],
                bbox[:, 3] * image_shape[0],
            ],
            axis=-1,
        )
        bbox = self.convert_to_xywh(bbox)
        return image, bbox, class_id


class AnchorBox:
    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]

        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2 ** i for i in range(3, 8)]
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self.compute_dims()

    def compute_dims(self):
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def get_anchors(self, feature_height, feature_width, level):
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims = tf.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

    def get_anchors(self, image_height, image_width):
        anchors = [
            self.get_anchors(
                tf.math.ceil(image_height / 2 ** i),
                tf.math.ceil(image_width / 2 ** i),
                i,
            )
            for i in range(3, 8)
        ]
        return tf.concat(anchors, axis=0)

class LabelEncoder:
    def __init__(self):
        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )
        self.utils = Utils()


    def _match_anchor_boxes(
        self, anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4
    ):
        iou_matrix = self.utils.compute_iou(anchor_boxes, gt_boxes)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),
        )

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis=-1,
        )
        box_target = box_target / self._box_variance
        return box_target

    def _encode_sample(self, image_shape, gt_boxes, cls_ids):
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
            anchor_boxes, gt_boxes
        )
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        cls_target = tf.where(
            tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids
        )
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)
        label = tf.concat([box_target, cls_target], axis=-1)
        return label

    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        images_shape = tf.shape(batch_images)
        batch_size = images_shape[0]

        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            label = self._encode_sample(images_shape, gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)
        batch_images = tf.keras.applications.resnet.preprocess_input(batch_images)
        return batch_images, labels.stack()