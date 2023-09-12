"""Executable script to apply bounding boxes to images."""

from pathlib import Path
import sys
import cv2
import numpy as np
import torch
from tqdm import tqdm
import os
from torch import jit


def prepare_image(image: np.ndarray, shape: tuple) -> np.ndarray:
    """Prepare the image for the network.

    Currently only supports the following
    - Resize
    - Normalise
    """
    # Normalize for Object Detection
    # print(image[0][0])
    # print("Equal ",image.array_equiv(image))
    # print("Equal ",(image == image).all())
    image_norm = cv2.normalize(
        src=image,
        dst=None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8UC1,
    )

    # print("Equal image_norm",(image == image_norm).all())

    # Resize input image to match net size
    image = cv2.resize(image, shape)

    # print(image)

    return image


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


def process_output(
    outputs: np.ndarray,
    size_ratio: np.ndarray,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> tuple[np.ndarray]:
    # outputs 
    # [ [x,y,w,h,p1,p2,p3...p80],   detection 1
    #   [x,y,w,h,p1,p2,p3...p80],   detection 2
    #   [x,y,w,h,p1,p2,p3...p80],   detection 3
    #   ...
    #   [x,y,w,h,p1,p2,p3...p80],   detection 80
    # ] 
 

    """Process the output of the yolo model to get a collection of bounding boxes.

    Parameters
    ----------
    outputs : np.ndarray
        Full YOLO model output of shape (8400, 84).
        8400 = Number of detected Objects
        84 = Box features (x_cent, y_cent, width, height, *prob_classes)
    size_ratio : np.ndarray
        The ratio of the original image size to the model input size.
    conf_threshold : float, optional
        The threshold for filtering out low confidence predictions.
        Default is 0.5.
    iou_threshold : float, optional
        The Intersection Over Union (IOU) threshold for non-maxima suppression.
        Default is 0.5.

    Returns
    -------
    tuple[np.ndarray]
        The surviving boxes after non-maxima suppression, their scores, and class IDs.
    """
    # outputs = outputs.numpy()

    def predicate(row):
        any_row_beats_threshold = (row > conf_threshold).any()
        return any_row_beats_threshold

    # print("outputs TYPE", type(outputs))
    # Ignore x,y,w,h, values
    obj_conf = outputs[:,4:].numpy()
    # Apply the predicate function to each row and create a boolean mask
    mask = np.apply_along_axis(predicate, axis=1, arr=obj_conf)
    print("mask", mask.shape)
    # outputs_above_conf_thresh
    outputs = outputs[mask]
    print("outputs", outputs.shape)
    outputs = outputs.numpy()
    print("outputs", outputs.shape)


    # Get the anchors the have a high confidence that they contain an object
    # Multiply class confidence with bounding box confidence    # valid_mask = max_scores > conf_threshold
    # outputs = outputs[valid_mask]
    # max_scores = max_scores[valid_mask]
    max_scores = np.max(outputs[:, 4:], axis=1)
    print("max_scores",max_scores)
    print("max_scores len ",max_scores.shape)

    # Filter out boxes where the class is ambiguous (low max score)
    # valid_mask = max_scores > conf_threshold
    # outputs = outputs[valid_mask]
    # max_scores = max_scores[valid_mask]

    # Get the class id corresponding to the max score
    class_ids = np.argmax(outputs[:, 4:], axis=-1)
    print("class_ids",class_ids)
    # np.set_printoptions(threshold=sys.maxsize)
    print("class_ids",class_ids)

    # Get bounding boxes for each object
    boxes = extract_boxes(outputs, size_ratio)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(
        boxes, max_scores, conf_threshold, iou_threshold)
    
    print("indices",indices)

    # Return the surviving boxes, their scores and class ids
    return boxes[indices], max_scores[indices], class_ids[indices]


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
    net = cv2.dnn.readNet(model_path)
    output_names = net.getUnconnectedOutLayersNames()

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

        image = cv2.imread(str(img_path))

        # Get the size ratio (width x height) to reposition the boxes
        size_ratio = np.divide(image.shape[1::-1], input_shape)

        # print("image", image[0][0:10])
        # print("image", image.shape)
        # Prepare the image and pass through the network
        input_shape = (640, 640)
        prepared_image = prepare_image(image, input_shape)
        # print("prepared:", prepared_image[0][0:10])
        # print("prepared:", prepared_image.shape)

        # PyTorch Method
        net = jit.load('./model/yolov8n.torchscript')
        net.eval()
        tensor = torch.from_numpy(prepared_image)
        tensor = tensor.transpose_(0, 2)
        # print("tensor Shape : ",tensor.shape)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.float()

        print("tensor Shape : ", tensor.shape)
        print(tensor[0])
        output_pt = net(tensor)
        # 
        print("output_pt", output_pt.shape)
        # x,y,w,h
        output_pt = output_pt.transpose_(1, 2)
        print("output_pt ", output_pt[0][0][0:20])
        # print("output_pt ", output_pt[0][1][0:20])
        print("output_pt ", output_pt.shape)

        # print("output_pt", output_pt.shape)

        # Get a list of bounding boxes in the original img dimensions
        boxes, scores, class_ids = process_output(
            output_pt[0], size_ratio, conf_threshold, iou_threshold
        )

        # # Draw the bounding boxes onto the original image and save
        output_img = draw_boxes(
            image, boxes, scores, class_ids, class_names, class_colors
        )
        cv2.imwrite(str(Path(output_folder, img_path.name)), output_img)


if __name__ == "__main__":
    main()
