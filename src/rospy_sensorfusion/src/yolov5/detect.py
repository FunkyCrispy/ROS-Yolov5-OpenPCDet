#! /usr/bin/env python
# coding=utf-8

### this script is based on detect_ori.py
### removed some currently needless lines

import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from yolov5.utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from yolov5.utils.plots import colors, plot_one_box
from yolov5.utils.torch_utils import select_device, load_classifier, time_sync

from yolov5.utils.augmentations import letterbox

import numba


### tried to accelerate this procedure, but no effect 
# @numba.jit
# def sem_process(det, im0, sem_img, names, save_img, line_thickness):
#     global class_names, class_colors

#     for x1, y1, x2, y2, conf, cls in det:
#         xyxy = x1, y1, x2, y2
#         # xyxy = [elem.cpu().numpy() for elem in xyxy]

#         if save_img:  # Add bbox to image
#             c = int(cls)  # integer class
#             label = names[c] if False else names[c] + str(conf)
#             plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)

#         # produce semantic image
#         x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
#         x1, y1, x2, y2 = max(0, x1), max(0, y1), min(im0.shape[1], x2), min(im0.shape[0], y2)
#         if class_names[int(cls)] == 'car' or class_names[int(cls)] == 'bus' or class_names[int(cls)] == 'truck':
#             sem_img[y1:y2+1, x1:x2+1, :] = class_colors['car']
#         elif class_names[int(cls)] == 'person' or class_names[int(cls)] == 'bicycle' or class_names[int(cls)] == 'motorcycle':
#             sem_img[y1:y2+1, x1:x2+1, :] = class_colors['person']

#     return sem_img, im0


@torch.no_grad()
def yolov5_inference(
        im0,
        model,
        weights='',  # model.pt path(s)
        source='',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, pt, onnx = False, w.endswith('.pt'), w.endswith('.onnx')  # inference type
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)

    # check img size
    imgsz = check_img_size(imgsz, s=stride)

    # letterbox function contains resize and padding operation
    img = letterbox(im0, imgsz, stride)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)  # continuous tensor when store in memory

    if pt:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
    elif onnx:
        img = img.astype('float32')
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    # Inference
    t1 = time_sync()
    if pt:
        visualize = None
        pred = model(img, augment=augment, visualize=visualize)[0]
    elif onnx:
        pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    det_rst = []
    det = pred[0]

    # global variables
    global class_names, class_colors, class_in_need

    ### det result process
    # filter out needless classes
    det_list = det.cpu().numpy().tolist()
    det_list = [elem for elem in det_list if class_names[int(elem[5])] in class_in_need]

    # put person objects behind with sort
    det_list = sorted(det_list, key=lambda x : class_names[int(x[5])] in ['person', 'bicycle', 'motorcycle'])
    det = torch.Tensor(np.array(det_list)).type_as(det)

    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    imc = im0.copy() if save_crop else im0  # for save_crop

    sem_img = np.zeros(im0.shape)

    if len(det):
        # Rescale boxes from img_size to im0 size, im0 size is the original image size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        # Write results
        # xyxy, conf, cls -- det
        for *xyxy, conf, cls in det:

            xyxy = [elem.cpu().numpy() for elem in xyxy]

            if save_img or save_crop or view_img:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)

            # produce semantic image
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(im0.shape[1], x2), min(im0.shape[0], y2)
            if class_names[int(cls)] == 'car' or class_names[int(cls)] == 'bus' or class_names[int(cls)] == 'truck':
                sem_img[y1:y2+1, x1:x2+1, :] = class_colors['car']
            elif class_names[int(cls)] == 'person' or class_names[int(cls)] == 'bicycle' or class_names[int(cls)] == 'motorcycle':
                sem_img[y1:y2+1, x1:x2+1, :] = class_colors['person']

        # det = det.cpu()
        # sem_img, im0 = sem_process(det, im0, sem_img, names, save_img, line_thickness)

    # sem_img is the semantic img, im0 is the original img
    return sem_img, im0


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/home/yolov5-chongqing/src/ctcc_detector/src/weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='/home/zhangli/data/UE09_labeled_data/ue09_2021-03-15-17-12-00/ue09_pylon02/0280.jpg', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt


# coco original class_name list, use with class_id
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']
class_colors = {'car' : [0, 0, 142], 'person' : [220, 20, 60], 'bus' : [0, 60, 100]}
class_in_need = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']


def main(img, opt):

    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    
    check_requirements(exclude=('tensorboard', 'thop'))

    _ = yolov5_inference(img, **vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    # need to provide img arr
    main(img, opt)