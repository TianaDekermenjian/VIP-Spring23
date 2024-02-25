import os
import cv2
import logging
import argparse
import numpy as np
from utils import YOLOv5s

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EdgeTPUModel")

parser = argparse.ArgumentParser("EdgeTPU test runner")

parser.add_argument("--model", "-m", help="Weights file", required=True)
parser.add_argument("--image", "-i", type=str, help="Image file to run detection on")
parser.add_argument("--labels", "-l", type=str, required=True, help="Labels file")
parser.add_argument("--display", "-d", action='store_true', help="Display detection on monitor")
parser.add_argument("--stream", "-s", action='store_true', help="Process video stream in real-time")
parser.add_argument("--conf", "-ct", type=float, default=0.5, help="Detection confidence threshold")
parser.add_argument("--iou", "-it", type=float, default=0.1, help="Detections IOU threshold")
parser.add_argument("--wb", "-b", type=int, default=10, help = "Weight of basketball")
parser.add_argument("--wp", "-p", type=int, default=7, help = "Weight of player")
args = parser.parse_args()

model = YOLOv5s(args.model, args.labels, args.conf, args.iou)

classes = model.load_classes(args.labels)

logger.info("Loaded {} classes".format(len(classes)))

if(args.image) is not None:
    logger.info("Testing on input image: {}".format(args.image))

    img = cv2.imread(args.image)

    input_image = model.preprocess_frame(args.image)

    output = model.inference(input_image)

    detections = model.postprocess(output)

    wt = 0

    height, width, _ = img.shape

    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection

        c = int(class_id)

        if c == 0:
            wt += args.wb
        elif c == 1:
            wt += args.wp

        label = f'{classes[c]} {conf:.2f}'
        weight = f"Weight of frame: {wt}"

        (text_width, text_height), baseline = cv2.getTextSize(weight, 0, 1, 2)
        text_x = int((width - text_width) / 2)
        text_y = text_height + 10

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), [0, 0, 255], 2)
        cv2.putText(img, label, (int(x1), int(y1-2)), 0, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        cv2.putText(img, weight, (text_x, text_y), 0, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    s = ""

    for c in np.unique(detections[:, -1]):
        n = (detections[:, -1] == c).sum()
        s += f"{n} {classes[int(c)]}{'s' * (n > 1)}, "

    if s != "":
        s = s.strip()
        s = s[:-1]

    logger.info("Detected: {}".format(s))

    filename, extension = os.path.splitext(args.image)
    output_filename = filename + "_result"
    output_path = output_filename + extension

    cv2.imwrite(output_path, img)

    if(args.display):
        cv2.imshow("Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
