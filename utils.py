import numpy as np
import torch
import cv2

import config


def intersection_over_union(box1, box2, box_format="midpoint", eps=1e-6):
    if box_format == "coco":
        w1 = box1[..., 2]
        h1 = box1[..., 3]
        w2 = box2[..., 2]
        h2 = box2[..., 3]
        box1_x1 = box1[..., 0]
        box1_x2 = box1[..., 0] + w1
        box1_y1 = box1[..., 1]
        box1_y2 = box1[..., 1] + h1
        box2_x1 = box2[..., 0]
        box2_x2 = box2[..., 0] + w2
        box2_y1 = box2[..., 1]
        box2_y2 = box2[..., 1] + h2
    if box_format == "midpoint":
        box1_x1 = box1[..., 0] - box1[..., 2] / 2
        box1_x2 = box1[..., 0] + box1[..., 2] / 2
        box1_y1 = box1[..., 1] - box1[..., 3] / 2
        box1_y2 = box1[..., 1] + box1[..., 3] / 2
        box2_x1 = box2[..., 0] - box2[..., 2] / 2
        box2_x2 = box2[..., 0] + box2[..., 2] / 2
        box2_y1 = box2[..., 1] - box2[..., 3] / 2
        box2_y2 = box2[..., 1] + box2[..., 3] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + eps)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
               or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
               < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def bbox_by_findContours(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.blur(hsv, (3, 3))
    hsv = cv2.Canny(hsv, 255 / 3, 255)
    hsv = cv2.dilate(hsv, None, iterations=1)
    hsv = cv2.erode(hsv, None, iterations=1)
    cnts, _ = cv2.findContours(hsv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if (area > 100):
            bboxes.append([x, y, w, h])
    return bboxes


def bbox_by_mser(img):
    bboxes = []
    mser = cv2.MSER_create(min_area=1000)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = img.copy()
    # detect regions in gray scale image
    regions, _ = mser.detectRegions(gray)

    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    for i, contour in enumerate(hulls):
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append([x, y, w, h])
    return bboxes


def bbox_by_selective_search(img):
    cv2.setUseOptimized(True)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    bboxes = []
    for i, rect in enumerate(rects):
        x, y, w, h = rect
        if (w * h > 100):
            bboxes.append([x, y, w, h])
    return bboxes


def get_true_bboxes(img, bboxes, model):
    transforms = config.test_transforms
    true_bboxes = []
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        possible_obj = img[y:y + h, x:x + w]
        possible_obj = transforms(image=possible_obj)["image"]
        possible_obj = torch.FloatTensor(possible_obj.unsqueeze(0)).to(config.DEVICE)
        out = model(possible_obj)
        pred, classes = torch.sigmoid(out).max(1)
        if not classes.item() == 0:
            true_bboxes.append([classes.item(), pred.item(), x, y, w, h])
            # cv2.imwrite("image"+str(i) + ".jpg", img[y:y+h,x:x+w])
    true_bboxes = non_max_suppression(true_bboxes, 0.1, 0.6, box_format="coco")
    return true_bboxes
