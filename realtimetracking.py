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
print(classes)

interpreter = etpu.make_interpreter(model_file)
interpreter.allocate_tensors()

# input tensor details
input_details = interpreter.get_input_details()

input_scale = input_details[0]['quantization'][0] #quantization parameter
input_zero = input_details[0]['quantization'][1]  #quantization parameter

logger.info("Input scale: {}".format(input_scale))
logger.info("Input zero-point: {}".format(input_zero))

input_size = common.input_size(interpreter) #input image size

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
old_size = img.shape[:2]
ratio = float(input_size[0]/max(old_size))
new_size = tuple([int(x*ratio) for x in old_size])

img_resized = cv2.resize(img, (new_size[1], new_size[0]))

# pad image
delta_w = input_size[0] - new_size[1]
delta_h = input_size[0] - new_size[0]
pad = (delta_w, delta_h)
color = [100, 100, 100]

im_padded = cv2.copyMakeBorder(img_resized, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT, value=color)

# prepare tensor
im_padded = im_padded.astype(np.float32)

min_value = np.min(im_padded)
max_value = np.max(im_padded)
new_min = -128
new_max = 127

#im_normalized = ((im_padded - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min
im_normalized = im_padded/255.0

if im_normalized.shape[0] == 3:
    im_normalized = im_normalized.transpose((1,2,0))

input_image = (im_normalized/input_scale) + input_zero
input_image = input_image[np.newaxis].astype(input_data_type)

interpreter.set_tensor(input_details[0]['index'], input_image)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]["index"])
output = (output.astype(np.float32) - output_zero) * output_scale

detections = []

for i in range(output.shape[2]):  # Iterate over the detections
    detection = {
        'xc': output[0, 0, i],
        'yc': output[0, 1, i],
        'width': output[0, 2, i],
        'height': output[0, 3, i],
        'class1_confidence': output[0, 4, i],
        'class2_confidence': output[0, 5, i],
    }
    detections.append(detection)

filtered_detections = []

for detection in detections:
    confidence1 = detection['class1_confidence']
    confidence2 = detection['class2_confidence']

    if confidence1>0.5 or confidence2 >0.5:
        filtered_detections.append(detection)

print(len(filtered_detections))

# scale coordinates according to image
pad_w, pad_h = pad
in_h, in_w = input_size
out_h, out_w, c = img.shape

ratio_w = out_w/(in_w - pad_w)
ratio_h = out_h/(in_h - pad_h)

color =  (255, 0, 0)
thickness = 3

for filtered_detection in filtered_detections:
    x1 = filtered_detection['xc']-filtered_detection['width']/2
    y1 = filtered_detection['yc']-filtered_detection['height']/2
    x2 = filtered_detection['xc']-filtered_detection['width']/2
    y2 = filtered_detection['yc']-filtered_detection['height']/2

    x1_scaled = int(x1*img_w)
    y1_scaled = int(y1*img_h)
    x2_scaled = int(x2*img_w)
    y2_scaled = int(y2*img_h)

    coord1 = max(0, x1_scaled)
    coord2 = max(0, y1_scaled)
    coord3 = min(img_w, x2_scaled)
    coord4 = min(img_h, y2_scaled)

    print(coord1)
    print(coord2)
    print(coord3)
    print(coord4)

    cv2.rectangle(img, (coord1, coord2), (coord3, coord4), color, thickness)

out = "result.jpg"
cv2.imwrite(out, img)
