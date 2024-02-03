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
model_file = os.path.join(script_dir, 'models-edgetpu/D1M2-full-integer-quant-new_edgetpu.tflite')
label_file = os.path.join(script_dir, 'labelmap.txt')
image_file = os.path.join(script_dir, 'test/test416.jpg')

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
im_normalized = im_padded/255.0

if im_normalized.shape[0] == 3:
    im_normalized = im_normalized.transpose((1,2,0))

input_image = (im_normalized/input_scale) + input_zero
input_image = input_image[np.newaxis].astype(input_data_type)

interpreter.set_tensor(input_details[0]['index'], input_image)
interpreter.invoke()

detections = (common.output_tensor(interpreter, 0).astype('float32') - output_zero) * output_scale

#detections = non_max_suppression(detections, 0.001, 0.1, None, False, 1000)

if(len(detections)):
    print("here")
    detections_scaled=[]

    in_h, in_w = input_size
    out_h, out_w, _ = img.shape

    ratio_w = out_w/(in_w - delta_w)
    ratio_h = out_h/(in_h - delta_h)

    for coord in detections[:,:4]:
        x1, y1, x2, y2 = coord
        x1 *= in_w*ratio_w
        x2 *= in_w*ratio_w
        y1 *= in_h*ratio_h
        y2 *= in_h*ratio_h

        x1 =np.maximum(0, x1)
        x2 =np.minimum(out_w, x2)

        y1 = np.maximum(0, y1)
        y2 = np.minimum(out_h, y2)

        detections_scaled.append((x1, y1, x2, y2))

    detections[:,:4] = np.array(detections_scaled).astype(int)

    output = {}

    s = ""

    detections[:, -1] = detections[:, -1].astype(int)

    for c in np.unique(detections[:, -1]):
        n = (detections[:, -1] == c).sum()
        s += f"{n} {classes[int(c)]}{'s' * (n > 1)}, "

    if s != "":
        s = s.strip()
        s = s[:-1]

    logger.info("Detected: {}".format(s))

