from keras.models import load_model
from objUtils import *


class Predict:
    def __init__(self, image):
        self.model = load_model(('model.h5'))
        self.image = image

    def predict(self):
        input_w, input_h = 416, 416
        photo_filename = self.image
        image, image_w, image_h = Utils.load_image_pixels(
            photo_filename, (input_w, input_h))
        yhat = self.model.predict(image)
        anchors = [[116, 90, 156, 198, 373, 326], [
            30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
        class_threshold = 0.6
        boxes = list()
        for i in range(len(yhat)):
            boxes += Utils.decode_netout(yhat[i][0],
                                         anchors[i], class_threshold, input_h, input_w)
        Utils.correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
        Utils.do_nms(boxes, 0.5)
        labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
                  "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                  "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                  "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                  "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                  "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                  "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        v_boxes, v_labels, v_scores = Utils.get_boxes(
            boxes, labels, class_threshold)
        '''
        for i in range(len(v_boxes)):
            #print(v_labels[i], v_scores[i])
            print(v_labels[i])
        '''
        Utils.draw_boxes(photo_filename, v_boxes, v_labels, v_scores)
        return ["Careful, " + v + " ahead." for v in v_labels]
