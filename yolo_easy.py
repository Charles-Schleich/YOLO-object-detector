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
    orig_shape: np.ndarray,
    box: np.ndarray,
    class_id: int,
    class_names: list,
) -> np.ndarray:
    """Draw the bounding boxes onto the pixel values of the image."""
    w_orig, h_orig = (orig_shape[0], orig_shape[1])
    x_1, y_1 = int(box[0]), int(box[1])
    x_2, y_2 = int(box[2]), int(box[3])

    print("class_id ", class_id, type(class_id))
    label = f"{class_names[class_id]} "
    # label = f"{class_names[class_id]} {int(score * 100)}%"
    scale = int(6)
    color = 6

    cv2.rectangle(image, (x_1, y_1), (x_2, y_2), color, thickness=scale)
    # cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, lw//5, lw//2)
    # cv2.rectangle(image, (0, 0), (x_2, y_2), 1, thickness=scale)
    cv2.putText(
        image,
        label,
        org=(x_1, y_1 - 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=scale//4,
        color=color,
        thickness=scale//2,
    )
    return image


image = cv2.imread(str(img1))
size_ratio = np.divide(image.shape[1::-1], (640, 640))
print("size_ratio ", size_ratio)

for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    names = result.names
    orig_shape = result.orig_shape

    # print(names)
    # xywh = boxes.xywh
    # classes = boxes.cls
    # print(boxes)
    # print(classes)

    for box in boxes:
        xyxy = box.xyxy
        cls = box.cls
        print(box.xyxy[0])
        print(box.xywh[0])
        print(box.xyxyn[0])
        print(box.xywhn[0])

        image = draw_box(image, orig_shape, xyxy[0], cls[0].item(), names)
        break

    # print(boxes.xywh)
cv2.imwrite(str(Path("./", "dog_out.png")), image)

# boxes
