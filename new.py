import os
import cv2
import time
import logging
import argparse
import numpy as np
from utils import YOLOv5s
import websockets
import asyncio
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EdgeTPUModel")

parser = argparse.ArgumentParser("EdgeTPU test runner")

parser.add_argument("--model", "-m", help="Weights file", required=True)
parser.add_argument("--labels", "-l", type=str, required=True, help="Labels file")
parser.add_argument("--display", "-d", action='store_true', help="Display detection on monitor")
parser.add_argument("--stream", "-s", action='store_true', help="Process video stream in real-time")
parser.add_argument("--device", "-dev", type=int, default=1, help="Camera to process feed from (0, for Coral Camera, 1 for USB")
parser.add_argument("--time", "-t", type = int, default = 300, help="Length of video to record")
parser.add_argument("--conf", "-ct", type=float, default=0.5, help="Detection confidence threshold")
parser.add_argument("--iou", "-it", type=float, default=0.1, help="Detections IOU threshold")
parser.add_argument("--wb", "-b", type=int, default=10, help = "Weight of basketball")
parser.add_argument("--wp", "-p", type=int, default=7, help = "Weight of player")
args = parser.parse_args()

model = YOLOv5s(args.model, args.labels, args.conf, args.iou)

classes = model.load_classes(args.labels)

logger.info("Loaded {} classes".format(len(classes)))

async def client():
    if (args.stream):

        logger.info("Opening stream on device: {}".format(args.device))

        cam = cv2.VideoCapture(args.device)

        start = time.time()
        start2 = time.time()

        fps = 20
        resolution = (752, 416)

        index = 0
        filename = f"/home/mendel/streams/video_{index}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(filename, fourcc, fps, resolution)

        wt = 0

        uri = "ws://172.20.10.4:6969/"  # Use your server's IP address and port
        flag = False
        
        async with websockets.connect(uri) as websocket:
            # Here you can add your logic to get the value you want to send
            # For demonstration, I'm using a random value
            
            value = str(random.random()) 
            await websocket.send(value)

            # Keep the connection open until a response is received
            while True:
                try:
                    # Receive the response from the server
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    if response=="start":
                        flag = True
                        break
                except asyncio.TimeoutError:
                    # If no response is received within the timeout, keep waiting
                    pass

            while time.time()-start< args.time:
                try:

                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=1)
                        print(response)
                    except asyncio.TimeoutError:
                        pass

                    res, frame = cam.read()

                    if res is False:
                        logger.error("Empty image received")
                        break
                    else:
                        input = model.preprocess_frame(frame)

                        output = model.inference(input)

                        detections = model.postprocess(output)

                        output_frame, weight = model.draw_bbox_weights(frame, detections, args.wb, args.wp)

                        wt += weight
                        wt += 2
                        writer.write(output_frame)

                        s = ""
#
  #                      for c in np.unique(detections[:, -1]):
 #                           n = (detections[:, -1] == c).sum()
   #                         s += f"{n} {classes[int(c)]}{'s' * (n > 1)}, "

    #                    if s != "":
     #                       s = s.strip()
      #                      s = s[:-1]

       #                 logger.info("Detected: {}".format(s))

                        if time.time()-start2 >=17:
                            await websocket.send(str(wt))
                            writer.release()
                            index += 1

                            wt = 0

                            filename = f"/home/mendel/streams/video_{index}.mp4"

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

                filename = f"/home/mendel/streams/video_{index}.mp4"
                index += 1

            cam.release()
            cv2.destroyAllWindows()


asyncio.get_event_loop().run_until_complete(client())
