import numpy as np
import cv2
import random

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("-", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def id2name(id):
    # # classes = dict()
    # with open("configs/coco_names.txt") as f:
    #     lines = f.read().splitlines()
    names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush']  # class names
    return names[int(id)]
    # return classes[int(id)]


def plot_bboxes(image, bboxes, identities=None, color=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    # tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    color_map = {"person": (60, 20, 220),
                 "car": (180, 105, 255),
                 "bus": (255, 191, 0),
                 "else": (50, 205, 50)
                 }

    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        label = '%s' % (id2name(cls_id))
        # color = compute_color_for_labels(0)
        # color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (x1, y1), (x2, y2)
        if label in color_map.keys():
            color = color_map[label]
        else:
            color = color_map["else"]
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 2, 1)  # font thickness
        text = label + " " + str(pos_id)
        t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{}'.format(text), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return image


def draw_person(img, bbox_xyxy, reid_results, names, identities=None, offset=(0, 0)):
    for i, x in enumerate(bbox_xyxy):
        person_name = names[reid_results[i]]
        t_size = cv2.getTextSize(person_name, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        color = compute_color_for_labels(0)
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        # 截取行人用来erson_bank.py获取行人特征
        '''
        p = img[int(x[1]):int(x[3]), int(x[0]):int(x[2])]
        cv2.imwrite('/home/zengwb/Documents/yolov5-fastreid/fast_reid/query/a1/a1.jpg', p)
        cv2.imshow('p', p)
        cv2.waitKey(0)
        '''
        cv2.rectangle(img, c1, c2, color, lineType=cv2.LINE_AA)
        cv2.rectangle(
            img, (c2[0] - t_size[0]-3, c2[1]-t_size[1] - 4), c2, color, -1)
        cv2.putText(img, person_name, (c2[0] - t_size[0]-3, c2[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        # cv2.rectangle(
        #     img, (c1[0] + 2*t_size[0], c1[1]), (c1[0] + 2*t_size[0] + 3, c1[1] + t_size[1] + 4), color, -1)
        # cv2.putText(img, person_name, (c1[0], c1[1] +
        #                          t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        # cv2.rectangle(img, (c2[0]+t_size[0] +3, c2[1]+t_size[1]+4), (c2[0]+2*t_size[0] +3, c2[1]+2*t_size[1]+4), color, -1)
        # cv2.putText(img, person_name, (c2[0]+t_size[0] +3, c2[1]+2*t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        # if label:
        #     tf = max(tl - 1, 1)  # font thickness
        #     t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        #     c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        #     cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        #     cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
