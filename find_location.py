import config
import torch
from utils import (bbox_by_mser as mser, bbox_by_findContours as fc, bbox_by_selective_search as ss, get_true_bboxes)
import matplotlib.pyplot as plt
import numpy as np
import cv2

TYPE = 2 #Работает только с selecive search


def get_bboxes(img):
    if TYPE == 0:
        bboxes = fc(img)
    elif TYPE == 1:
        bboxes = mser(img)
    elif TYPE == 2:
        bboxes = ss(img)
    return bboxes


def main():
    model = torch.load(config.MODEL_FILENAME).to(config.DEVICE)
    image = cv2.imread("test_images/Apple 49.png")
    bboxes = get_bboxes(image)
    true_bboxes = get_true_bboxes(img=image, bboxes=bboxes, model=model)
    visualize_bbox(image, true_bboxes)


def visualize_bbox(img, bboxes, color=config.BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    for bbox in bboxes:
        classes, p, x, y, w, h = bbox
        x_min, x_max, y_min, y_max = int(x), int(x + w), int(y), int(
            y + h)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

        ((text_width, text_height), _) = cv2.getTextSize(config.classes[classes], cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), config.BOX_COLOR, 1)
        cv2.putText(
            img,
            text=config.classes[classes] + " " + str(round(p, 2)),
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=config.TEXT_COLOR,
            lineType=cv2.LINE_AA,
        )

    # plt.figure(figsize=(12, 12))
    # plt.axis('off')
    # plt.imshow(img)
    cv2.imshow("image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    #plt.show()


if __name__ == "__main__":
    main()
