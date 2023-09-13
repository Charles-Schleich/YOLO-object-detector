import argparse

import cv2.dnn
import numpy as np
import torch

from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml

CLASSES = yaml_load(check_yaml('coco128.yaml'))['names']

colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def main(onnx_model, input_image):
    input_image = "./images/dog.png"
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)
    original_image: np.ndarray = cv2.imread(input_image)
    [height, width, _] = original_image.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / 640

    image = cv2.resize(image, (640, 640))
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("input_image:", image[0][0:3], image[0][-1])
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, swapRB=True)
    # blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)

    print("type blob:", type(blob))
    print("input blob:", blob.shape)
    blob2 = blob[0]
    # print(blob2.shape)
    # print(blob2[0])
    # print(blob2[0][0][639])

    model.setInput(blob)
    outputs = model.forward()
    # print("outputs.shape",outputs.shape)
    # print("outputs ",outputs)
    outputs = np.array(cv2.transpose(outputs[0]))
    # rows = outputs.squeeze()
    # print("outputs.shape",outputs.shape)
    print("outputs.type",type(outputs))
    print("outputs.type",outputs.dtype)
    print(outputs[0][0:6])

    rows = outputs.shape[0]
    ()
    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[i][0] - (0.5 * outputs[i][2]), outputs[i][1] - (0.5 * outputs[i][3]),
                outputs[i][2], outputs[i][3]]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
    # print(result_boxes)

    detections = []
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            'class_id': class_ids[index],
            'class_name': CLASSES[class_ids[index]],
            'confidence': scores[index],
            'box': box,
            'scale': scale}
        detections.append(detection)
        draw_bounding_box(original_image, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
                          round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))

    cv2.imshow('image', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8n.onnx', help='Input your onnx model.')
    parser.add_argument('--img', default=str(ASSETS / 'bus.jpg'), help='Path to input image.')
    args = parser.parse_args()
    main(args.model, args.img)