from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

img1 = './images/dog.png'
# Run batched inference on a list of images
results = model([img1])  # return a list of Results objects

# Process results list


def draw_box(
    image: np.ndarray,
    box: np.ndarray,
    class_id: int,
    class_names: list,
) -> np.ndarray:
    """Draw the bounding boxes onto the pixel values of the image."""
    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    print("class_id ", class_id, type(class_id))
    label = f"{class_names[class_id]} "
    # label = f"{class_names[class_id]} {int(score * 100)}%"
    scale = int(6)
    color = 6

    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=scale)
    # cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, lw//5, lw//2)
    cv2.putText(
        image,
        label,
        org=(x, y - 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=scale//4,
        color=color,
        thickness=scale//2,
    )
    return image


for result in results:
    # print(result)
    boxes = result.boxes  # Boxes object for bbox outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs
    names = result.names
    print(names)

    xywh = boxes.xywh
    classes = boxes.cls
    # print(boxes)
    # print(classes)
    image = cv2.imread(str(img1))
    for box in boxes:
        xywh = box.xywh
        cls = box.cls
        print(xywh)
        image = draw_box(image, xywh[0], cls[0].item(), names)
        # print(box)

    # print(boxes.xywh)
cv2.imwrite(str(Path("./", "dog_out.png")), image)

# boxes
