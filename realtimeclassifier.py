import os
import cv2
import time
import logging
import argparse
import numpy as np
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EdgeTPUModel")

parser = argparse.ArgumentParser("Classifier test")

parser.add_argument("--model", "-m", help="Weights file", required=True)
parser.add_argument("--image", "-i", type=str, help="Image file to run detection on")
parser.add_argument("--labels", "-l", type=str, required=True, help="Labels file")
parser.add_argument("--display", "-d", action='store_true', help="Display detection on monitor")
parser.add_argument("--stream", "-s", action='store_true', help="Process video stream in real-time")
parser.add_argument("--device", "-dev", type=int, default=1, help="Camera to process feed from (0, for Coral Camera, 1 for USB")
parser.add_argument("--time", "-t", type = int, default = 300, help="Length of video to record")
parser.add_argument("--conf", "-ct", type=float, default=0.5, help="Detection confidence threshold")
parser.add_argument("--iou", "-it", type=float, default=0.1, help="Detections IOU threshold")
args = parser.parse_args()

interpreter = edgetpu.make_interpreter(args.model)
interpreter.allocate_tensors()

if(args.image) is not None:
    logger.info("Testing on input image: {}".format(args.image))

    img = cv2.imread(args.image)

    size = common.input_size(interpreter)

    print(size)

    img_resized = cv2.resize(img, size)

