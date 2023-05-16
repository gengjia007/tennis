import argparse
import datetime
import time
from pathlib import Path
import json
import torch

from models.common import DetectMultiBackend
from mouse_controller import MouseController
from screenshot import ScreenShot
from utils.dataloaders import LoadImages
from utils.general import (Profile, check_img_size, non_max_suppression, print_args, scale_boxes)
from utils.torch_utils import select_device


class Tennis:
    def __init__(self,
                 weights='yolov5s.pt',  # model path or triton URL
                 puzzle_weights='yolov5s.pt',
                 source='data/images',  # file/dir/URL/glob/screen/0(webcam)
                 puzzle_source='data/images',
                 position_json='position.json',
                 data='data/coco128.yaml',  # dataset.yaml path
                 imgsz=(640, 640),  # inference size (height, width)
                 conf_thres=0.25,  # confidence threshold
                 iou_thres=0.45,  # NMS IOU threshold
                 max_det=1000,  # maximum detections per image
                 device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 classes=None,  # filter by class: --class 0, or --class 0 2 3
                 agnostic_nms=False,  # class-agnostic NMS
                 augment=False,  # augmented inference
                 visualize=False,  # visualize features
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
        self.po = json.load(open(position_json))

        # Load model
        device = select_device(device)
        self.model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        self.puzzle_model = DetectMultiBackend(puzzle_weights, device=device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

    def inference(self, model, source):
        # Dataloader
        bs = 1  # batch_size
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt,
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
                pred = model(im, augment=self.augment, visualize=self.visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                           max_det=self.max_det)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                s += '%gx%g ' % im.shape[2:]  # print string
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    return det
        return None

    def run(self):
        self.mc.move_and_click((self.po["enter"][0] + self.po["tennis_window"][0], self.po["enter"][1] + self.po["tennis_window"][1]))
        # time.sleep(0.1)
        self.mc.move_and_click((self.po["indoor"][0] + self.po["tennis_window"][0], self.po["indoor"][1] + self.po["tennis_window"][1]))
        # time.sleep(0.1)
        last_time = (self.po["last_time"][0] + self.po["tennis_window"][0], self.po["last_time"][1] + self.po["tennis_window"][1])
        self.mc.move(last_time)
        self.mc.hscroll(-500)
        self.mc.double_click()
        # time.sleep(0.1)
        self.mc.move((last_time[0], last_time[0]+65))
        self.mc.vscroll(-500)
        ss = ScreenShot(self.po["tennis_window"], "./screenshot/area/")
        ss.run()
        det = self.inference(self.model, self.source)
        if det is not None:
            price = []
            submit = []
            for item in det.numpy():
                if item[-1] == 0:
                    price.append([(item[0] + item[2]) / 2, (item[1] + item[3]) / 2])
                elif item[-1] == 1:
                    submit.append([(item[0] + item[2]) / 2, (item[1] + item[3]) / 2])
            price.sort(key=lambda k: k[1])
            final_price = price
            final_price = [[item[0] + self.po["tennis_window"][0], item[1] + self.po["tennis_window"][1]] for item in final_price]
            submit = [[item[0] + self.po["tennis_window"][0], item[1] + self.po["tennis_window"][1]] for item in submit]
            if len(final_price) != 0:
                # final_price = [item for item in final_price if item[1] > 433]
                print(
                    "检测到{}个空闲时间, 坐标分别为:{}, 检测到{}个提交按钮，坐标为{}".format(len(final_price),
                                                                                            str(final_price),
                                                                                            len(submit),
                                                                                            str(submit)))
                final_price.sort(key=lambda k: (k[1], -k[0]), reverse=True)
                self.mc.move_and_single_click(final_price[0])
                print("first:{}".format(final_price[0]))
                for item in final_price[1:]:
                    if abs(item[1] - final_price[0][1]) > 20:
                        self.mc.move_and_single_click(item)
                        print("second:{}".format(item))
                        break
                self.mc.move_and_single_click((self.po["submit_button"][0] + self.po["tennis_window"][0], self.po["submit_button"][1] + self.po["tennis_window"][1]))
                time.sleep(1.5)
                ss = ScreenShot(self.po["tennis_window"], "./screenshot/puzzle/")
                ss.run()

                det = self.inference(self.puzzle_model, self.puzzle_source)
                if det is not None:
                    a = 0
                    b = 0
                    for item in det.numpy():
                        if item[0] > 100:
                            if item[0] > b:
                                b = item[0]
                        else:
                            if item[0] > a:
                                a = item[0]
                    self.mc.move((a, self.po["puzzle_button_y"] + self.po["tennis_window"][1]))
                    self.mc.drag((b, self.po["puzzle_button_y"] + self.po["tennis_window"][1]))
            else:
                print("未检测到目标区域，请到路径{}查看截图".format("./screenshot/area/ss.png"))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--puzzle-weights', nargs='+', type=str, default='yolov5s.pt',
                        help='model path or triton URL')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--puzzle-source', type=str, default='data/images',
                        help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--position-json', type=str, default='position.json', help='')
    parser.add_argument('--data', type=str, default='data/tennis.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    model = Tennis(**vars(opt))
    lighting_time = "12:00"
    print("waiting {} to run".format(lighting_time))
    while True:
        time.sleep(0.01)
        current_time = datetime.datetime.now()
        if str(current_time.time()).startswith(lighting_time):
            print("current_time:    " + str(current_time.time()))
            break
    model.run()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
