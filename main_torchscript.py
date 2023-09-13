"""Executable script to apply bounding boxes to images."""

from pathlib import Path
import sys
import cv2
import numpy as np
import torch
from tqdm import tqdm
import os
from torch import jit
import matplotlib.pyplot as plt


def forward_pass(
    image: np.ndarray, network: cv2.dnn.Net, output_names: tuple
) -> np.ndarray:
    """Pass a pre-processed image through the network and extract the outputs.

    Will also apply 1/255 scaling to the input image
    """
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0)
    network.setInput(blob)
    # 1 for output type, 1 for batch dim
    return network.forward(output_names)[0][0]


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes):
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    label = f'{classes[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_names: list,
    class_colors: np.ndarray,
) -> np.ndarray:
    """Draw the bounding boxes onto the pixel values of the image."""
    for box, score, class_id in zip(boxes, scores, class_ids):
        # Get the pixes coordinates and the appropriate colour and name
        x, y, w, h = box.astype(int)
        color = class_colors[class_id]
        label = f"{class_names[class_id]} {int(score * 100)}%"

        # TODO Find a better way to perform the autoscaling
        scale = int(0.005 * min(image.shape[:2]))

        # Draw the rectangle and text onto the image
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=scale)
        # cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, lw//5, lw//2)
        cv2.putText(
            image,
            label,
            org=(x, y - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=scale // 4,
            color=color,
            thickness=scale // 2,
        )
    return image


def extract_boxes(predictions: np.ndarray, size_ratio: np.ndarray) -> np.ndarray:
    """Pull out the bounding boxes from the predictions and resize to original."""

    # Extract boxes from predictions (x_center, y_center, width, height)
    boxes = predictions[:, :4]

    # Scale boxes to original image dimensions
    boxes *= np.array([*size_ratio, *size_ratio])

    # Change the format to (x_left, y_bottom, width, height)
    boxes[..., 0] -= boxes[..., 2] * 0.5
    boxes[..., 1] -= boxes[..., 3] * 0.5
    return boxes


def main() -> None:
    """Run script."""

    # Initial args
    model_path = "model/yolov7-tiny_480x640.onnx"
    class_name_path = "model/class_names.txt"
    image_folder = "images"
    output_folder = "outputs"
    conf_threshold = 0.5
    iou_threshold = 0.5

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the pretrained network ONNX file and the associated class names
    # net = cv2.dnn.readNet(model_path)
    # output_names = net.getUnconnectedOutLayersNames()

    # Get the expected image dimensions based on the model name (width x height)
    input_shape = Path(model_path).stem.split("_")[-1].split("x")
    input_shape = [int(x) for x in input_shape]
    input_shape = input_shape[::-1]

    # Get names and colors to represent each class
    class_names = [x.strip() for x in open(class_name_path).readlines()]
    class_colors = np.random.default_rng(3).uniform(
        0, 255, size=(len(class_names), 3))

    # Get a list of images to run over
    file_list = list(Path(image_folder).glob("*dog*"))
    # file_list = list(Path(image_folder).glob("*"))
    for img_path in tqdm(file_list, "Drawing boxes on images"):
        print(img_path)

        original_image = cv2.imread(str(img_path))
        [height, width, _] = original_image.shape
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image
        scale = length / 640

        # Prepare the image and pass through the network
        input_shape = (640, 640)
        image = cv2.resize(image, input_shape)
        print("image type:", type(image))
        print("image")
        print(image[0][0:3], image[0][-1])

        # PyTorch Method
        net = jit.load('./yolov8n.torchscript')
        tensor = torch.from_numpy(image)
        # Transpose height and width
        tensor = tensor.transpose_(0, 1)
        tensor = tensor.transpose_(0, 2)
        print("input tensor", tensor.shape)
        # Swap BGR -> RGB
        tensor = tensor[[2, 1, 0]]
        tensor = tensor.float()
        tensor = tensor/255
        tensor = tensor.unsqueeze(0)
        # Pass through Network
        output = net(tensor)
        output = output.transpose_(1, 2)
        output = np.array(output)
        output = output.squeeze()
        # print("output")
        # print("output", output.shape)
        # print("outputs.type",type(output))
        # print("outputs.type",output.dtype)
        # print(output[0][0:6])
        rows = output.shape[0]
        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            classes_scores = output[i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)
             ) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [
                    output[i][0] - (0.5 * output[i][2]
                                    ), output[i][1] - (0.5 * output[i][3]),
                    output[i][2], output[i][3]]
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
                'class_name': class_names[class_ids[index]],
                'confidence': scores[index],
                'box': box,
                'scale': scale}
            detections.append(detection)
            draw_bounding_box(original_image, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
                              round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale), class_names)

        cv2.imshow('image', original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return detections


if __name__ == "__main__":
    main()
