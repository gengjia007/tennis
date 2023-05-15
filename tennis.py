import argparse
import datetime
import time
from pathlib import Path

import torch

from models.common import DetectMultiBackend
from mouse_controller import MouseController
from screenshot import ScreenShot
from utils.dataloaders import LoadImages
from utils.general import (Profile, check_img_size, check_requirements, non_max_suppression, print_args, scale_boxes)
from utils.torch_utils import select_device


class Tennis:
    def __init__(self,
                 weights='yolov5s.pt',  # model path or triton URL
                 puzzle_weights='yolov5s.pt',
                 source='data/images',  # file/dir/URL/glob/screen/0(webcam)
                 puzzle_source='data/images',
                 data='data/coco128.yaml',  # dataset.yaml path
                 imgsz=(640, 640),  # inference size (height, width)
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
                 dnn=False,  # use OpenCV DNN for ONNX inference
                 vid_stride=1  # video frame-rate stride
                 ):
        self.source = str(source)
        self.puzzle_source = str(puzzle_source)
        self.vid_stride = vid_stride
        self.augment = augment
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms
        self.classes = classes
        self.visualize = visualize
        self.mc = MouseController()
        self.button = 523 + 37

        # Load model
        device = select_device(device)
        self.model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        self.puzzle_model = DetectMultiBackend(puzzle_weights, device=device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

    def run(self):
        while True:
            self.mc.move_and_click((320, 770))
            time.sleep(0.5)
            self.mc.move_and_click((209, 413 + 37))
            time.sleep(0.2)
            self.mc.move((368, 492 + 37))
            self.mc.hscroll(-500)
            self.mc.move_and_click((361, 545))
            self.mc.scroll(-500)
            time.sleep(0.2)
            shift_x = 0 + 37
            ss = ScreenShot((0, shift_x, 413, 782), "./screenshot/area/")
            ss.run()
            # Dataloader
            bs = 1  # batch_size
            dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt,
                                 vid_stride=self.vid_stride)

            # Run inference
            self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *self.imgsz))  # warmup
            seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
            for path, im, im0s, vid_cap, s in dataset:
                with dt[0]:
                    im = torch.from_numpy(im).to(self.model.device)
                    im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim

                # Inference
                with dt[1]:
                    # self.visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.visualize else False
                    pred = self.model(im, augment=self.augment, visualize=self.visualize)

                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                               max_det=self.max_det)

                # Second-stage classifier (optional)
                # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                price = []
                submit = []
                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)  # to Path
                    s += '%gx%g ' % im.shape[2:]  # print string
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                        for item in det.numpy():
                            if item[-1] == 0:
                                price.append([(item[0] + item[2]) / 2, (item[1] + item[3]) / 2])
                            elif item[-1] == 1:
                                submit.append([(item[0] + item[2]) / 2, (item[1] + item[3]) / 2])
                price.sort(key=lambda k: k[1])
                print(price)
                '''
                price.append([1000, 500])
                final_price = []
                final_price.append(price[0])
                for i in range(1, len(price)):
                    if price[i][0] - price[i - 1][0] >= 20:
                        final_price.append(price[i - 1])
                '''
                final_price = price
                final_price = [[item[0], item[1] + shift_x] for item in final_price]
                submit = [[item[0], item[1] + shift_x] for item in submit]
                if len(final_price) != 0:
                    # final_price = [item for item in final_price if item[1] > 469]
                    print(
                        "检测到{}个空闲时间, 坐标分别为:{}, 检测到{}个提交按钮，坐标为{}".format(len(final_price),
                                                                                                str(final_price),
                                                                                                len(submit),
                                                                                                str(submit)))
                    final_price.sort(key=lambda k: k[1])
                    self.mc.move_and_single_click(final_price[0])
                    print("first:{}".format(final_price[0]))
                    for item in final_price[1:]:
                        if item[1] - final_price[0][1] > 20:
                            self.mc.move_and_single_click(item)
                            print("second:{}".format(item))
                            break
                    self.mc.move_and_single_click((337, 718 + 37))
                    # exit()
                    time.sleep(1)
                    ss = ScreenShot((0, 37, 413, 782), "./screenshot/puzzle/")
                    ss.run()

                    bs = 1  # batch_size
                    dataset = LoadImages(self.puzzle_source, img_size=self.imgsz, stride=self.stride, auto=self.pt,
                                         vid_stride=self.vid_stride)

                    # Run inference
                    self.puzzle_model.warmup(
                        imgsz=(1 if self.pt or self.puzzle_model.triton else bs, 3, *self.imgsz))  # warmup
                    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
                    for path, im, im0s, vid_cap, s in dataset:
                        with dt[0]:
                            im = torch.from_numpy(im).to(self.puzzle_model.device)
                            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                            im /= 255  # 0 - 255 to 0.0 - 1.0
                            if len(im.shape) == 3:
                                im = im[None]  # expand for batch dim

                        # Inference
                        with dt[1]:
                            # self.visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.visualize else False
                            pred = self.puzzle_model(im, augment=self.augment, visualize=self.visualize)

                        # NMS
                        with dt[2]:
                            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes,
                                                       self.agnostic_nms,
                                                       max_det=self.max_det)

                        # Second-stage classifier (optional)
                        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
                        a = 0
                        b = 0
                        # Process predictions
                        for i, det in enumerate(pred):  # per image
                            seen += 1
                            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                            p = Path(p)  # to Path
                            s += '%gx%g ' % im.shape[2:]  # print string
                            if len(det):
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                            for item in det.numpy():
                                if item[0] > 100:
                                    if item[0] > b:
                                        b = item[0]
                                else:
                                    if item[0] > a:
                                        a = item[0]
                        self.mc.move((a, 557))
                        self.mc.drag((b, 557))
            break


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--puzzle-weights', nargs='+', type=str, default='yolov5s.pt',
                        help='model path or triton URL')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--puzzle-source', type=str, default='data/images',
                        help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default='data/tennis.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
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
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    model = Tennis(**vars(opt))

    while True:
        time.sleep(0.01)
        current_time = datetime.datetime.now()
        if str(current_time.time()).startswith("14:51"):
            print("current_time:    " + str(current_time.time()))
            break

    model.run()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
