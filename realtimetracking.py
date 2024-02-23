import os
import cv2
import yaml
import pathlib
import logging
import numpy as np
import pycoral.utils.edgetpu as etpu
from pycoral.adapters import common
from nms import non_max_suppression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EdgeTPUModel")

script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'models-edgetpu/yolov5s-224-D1_edgetpu.tflite')
label_file = os.path.join(script_dir, 'labelmap.txt')
image_file = os.path.join(script_dir, 'image2.jpg')

with open(label_file, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

# load classes
classes = cfg['names']
logger.info("Loaded {} classes".format(len(classes)))

interpreter = etpu.make_interpreter(model_file)
interpreter.allocate_tensors()

# input tensor details
input_details = interpreter.get_input_details()

input_scale = input_details[0]['quantization'][0] #quantization parameter
input_zero = input_details[0]['quantization'][1]  #quantization parameter

logger.info("Input scale: {}".format(input_scale))
logger.info("Input zero-point: {}".format(input_zero))

input_size = common.input_size(interpreter) #input tensor size

logger.info("Image size: {}".format(input_size))

input_data_type = input_details[0]['dtype'] #input data type
logger.info("Expected input data type: {}". format(input_data_type))

# output tensor details
output_details = interpreter.get_output_details()

output_scale = input_details[0]['quantization'][0] #quantization parameter
output_zero = input_details[0]['quantization'][1]  #quantization parameter

logger.info("Output scale: {}".format(output_scale))
logger.info("Output zero-point: {}".format(output_zero))

output_data_type = output_details[0]['dtype'] #output data type
logger.info("Expected output data type: {}". format(output_data_type))

# process image
img = cv2.imread(image_file)

img_h, img_w, c = img.shape

# resize image
original_image_size = img.shape[:2]
ratio = float(input_size[0]/max(original_image_size))
new_size = tuple([int(x*ratio) for x in original_image_size])

img_resized = cv2.resize(img, (new_size[1], new_size[0]))

# pad image
pad_w = input_size[0] - new_size[1]
pad_h = input_size[0] - new_size[0]
pad = (pad_w, pad_h)
color = [100, 100, 100]

im_padded = cv2.copyMakeBorder(img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=color)

# prepare tensor
im_padded = im_padded.astype(np.float32)

im_normalized = im_padded/255.0

if im_normalized.shape[0] == 3:
    im_normalized = im_normalized.transpose((1,2,0))

input_image = (im_normalized/input_scale) + input_zero
input_image = input_image[np.newaxis].astype(input_data_type)

interpreter.set_tensor(input_details[0]['index'], input_image)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]["index"])
output = (output.astype(np.float32) - output_zero) * output_scale

nms_result = non_max_suppression(output, 0.5, 0, None, False, 10000)

detections = nms_result[0]

ratio_w = img_w/(input_size[0] - pad_w)
ratio_h = img_h/(input_size[1] - pad_h)

scaled_coordinates = []

if len(detections):
    for coordinates in detections[:,:4]:
        x1, y1, x2, y2 = coordinates

        x1_scaled = max(0, int((x1*input_size[0]*ratio_w)))
        y1_scaled = max(0, int((y1*input_size[1]*ratio_h)))
        x2_scaled = min(int(img_w), int((x2*input_size[0]*ratio_w)))
        y2_scaled = min(int(img_h) ,int((y2*input_size[1]*ratio_h)))

        scaled_coordinates.append((x1_scaled, y1_scaled, x2_scaled, y2_scaled))

        cv2.rectangle(img, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), [100, 100, 100], 2)

    cv2.imshow("Detections:", img)

    if cv2.waitKey(1) == ord('q'):
        break

    detections[:,:4] = scaled_coordinates

    s = ""

    for c in np.unique(detections[:, -1]):
        n = (detections[:, -1] == c).sum()
        s += f"{n} {classes[int(c)]}{'s' * (n > 1)}, " 

    if s != "":
        s = s.strip()
        s = s[:-1]

    logger.info("Detected: {}".format(s))

cv2.destroyAllWindows()
