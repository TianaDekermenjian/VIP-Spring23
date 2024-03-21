import os
import cv2
import time
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
parser.add_argument("--device", "-dev", type=int, default=1, help="Camera to process feed from (0, for Coral Camera, 1 for USB")
parser.add_argument("--time", "-t", type = int, default = 300, help="Length of video to record")
parser.add_argument("--conf", "-ct", type=float, default=0.5, help="Detection confidence threshold")
parser.add_argument("--iou", "-it", type=float, default=0.1, help="Detections IOU threshold")
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

    output_img = model.draw_bbox(img, detections)

    if len(detections):
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

    cv2.imwrite(output_path, output_img)

    if(args.display):
        cv2.imshow("Detection", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

elif (args.stream):
    logger.info("Opening stream on device: {}".format(args.device))

    cam = cv2.VideoCapture(args.device)

    start = time.time()
    start2 = time.time()

    fps = 20
    resolution = (640, 480)

    index = 0
    filename = f"/home/mendel/VIP/VIP-Spring23/streams/new_video_{index}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filename, fourcc, fps, resolution)

    while time.time()-start< args.time:
        try:
            res, frame = cam.read()

            if res is False:
                logger.error("Empty image received")
                break

            else:
                input = model.preprocess_frame(frame)

                output = model.inference(input)

                detections = model.postprocess(output)

                output_frame = model.draw_bbox(frame, detections)

                writer.write(output_frame)

                if len(detections):
                    s = ""

                    for c in np.unique(detections[:, -1]):
                        n = (detections[:, -1] == c).sum()
                        s += f"{n} {classes[int(c)]}{'s' * (n > 1)}, "

                    if s != "":
                        s = s.strip()
                        s = s[:-1]

                    logger.info("Detected: {}".format(s))

                if time.time()-start2 >=17:
                    writer.release()
                    index += 1

                    filename = f"/home/mendel/VIP/VIP-Spring23/streams/new_video_{index}.mp4"

                    writer = cv2.VideoWriter(filename, fourcc, fps, resolution)

                    start2 = time.time()

                cv2.waitKey(1)

                if(args.display):
                    cv2.imshow("Detection", output_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

        except KeyboardInterrupt:
            break

    else:
        writer.release()

        filename = f"/home/mendel/VIP/VIP-Spring23/streams/new_video_{index}.mp4"
        index += 1

    cam.release()
    cv2.destroyAllWindows()

