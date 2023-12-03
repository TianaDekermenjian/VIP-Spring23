TEST_FILE = r'/home/mendel/VIP/VIP-Spring23/test/'
TF_LITE_MODEL = r'/home/mendel/VIP/VIP-Spring23/models (edgetpu)/paul_best-int8_edgetpu.tflite'
LABEL_MAP = r'sunnylabels.txt'
BOX_THRESHOLD = 0.5
CLASS_THRESHOLD = 0.5
LABEL_SIZE = 0.5

import os
import cv2
import numpy as np
import random
import tflite_runtime.interpreter as tflite
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify

# Check if the files exist
if not os.path.exists(TEST_FILE):
    raise FileNotFoundError(f'Test file not found: {TEST_FILE}')
if not os.path.exists(TF_LITE_MODEL):
    raise FileNotFoundError(f'Model file not found: {TF_LITE_MODEL}')
if not os.path.exists(LABEL_MAP):
    raise FileNotFoundError(f'Label map file not found: {LABEL_MAP}')

interpreter = edgetpu.make_interpreter(TF_LITE_MODEL)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details() 
output_details = interpreter.get_output_details() 

_, height, width, _ = interpreter.get_input_details()[0]['shape']
input_scale, input_zero_point = input_details[0]['quantization']
input_type = input_details[0]['dtype']
output_type = output_details[0]['dtype']
output_scale, output_zero_point = output_details[0]['quantization'] 

with open(LABEL_MAP, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

image_files = [f for f in os.listdir(TEST_FILE) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Check if there are any image files
if not image_files:
    raise ValueError('No image files found')

image_file = random.choice(image_files)
image_path = os.path.join(TEST_FILE, image_file)

img = cv2.imread(image_path, cv2.IMREAD_COLOR)
IMG_HEIGHT, IMG_WIDTH = img.shape[:2]

pad = round(abs(IMG_WIDTH - IMG_HEIGHT) / 2) 
x_pad = pad if IMG_HEIGHT > IMG_WIDTH else 0
y_pad = pad if IMG_WIDTH > IMG_HEIGHT else 0
img_padded = cv2.copyMakeBorder(img, top=y_pad, bottom=y_pad, left=x_pad, right=x_pad,
                                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
IMG_HEIGHT, IMG_WIDTH = img_padded.shape[:2]
 
img_resized = cv2.resize(img_padded, (width, height), interpolation=cv2.INTER_AREA)

img_scaled = (img_resized/input_scale) + input_zero_point
img_scaled = np.around(img_scaled)

input_data = np.expand_dims(img_scaled, axis=0).astype(input_type) 

common.set_input(interpreter, input_data) 
interpreter.invoke()

outputs = interpreter.get_tensor(output_details[0]['index'])[0]
outputs = output_scale * (outputs.astype(np.float32) - output_zero_point) 

i = outputs.argmax()
print(i)
prediction = []
prediction.append(outputs[i])

print(prediction)

boxes = []
box_confidences = []
classes = []
class_probs = []

box_confidence = prediction[0][4] 

class_ = prediction[0][5:].argmax(axis=0)
class_prob = prediction[0][5:][class_]
 

cx, cy, w, h = prediction[0][:4] * np.array([IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH, IMG_HEIGHT])
x = round(cx - w / 2)
y = round(cy - h / 2)
w, h = round(w), round(h)

boxes.append([x, y, w, h])
box_confidences.append(box_confidence)
classes.append(class_)
class_probs.append(class_prob)

print(len(boxes))

#for i in range(len(boxes)):
x, y, w, h = boxes[0]
x, y, w, h = map(int, [x, y, w, h])

score = box_confidences[0]*class_probs[0]
class_name = classes[0]

text_color = (255, 0, 0)

cv2.rectangle(img_resized, (x, y), (x + w, y + h), text_color, 2)

label = f'{class_name}: {score * 100:.2f}%'
labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SIZE, 2)
cv2.rectangle(img_resized,
             (x, y + baseLine), (x + labelSize[0], y - baseLine - labelSize[1]),
             text_color, cv2.FILLED)
cv2.putText(img_resized, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, LABEL_SIZE, text_color, 1)

img_show = img_resized[y_pad: IMG_HEIGHT - y_pad, x_pad: IMG_WIDTH - x_pad]
cv2.namedWindow('Object detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object detection',
                1024 if IMG_WIDTH > IMG_HEIGHT else round(1024 * IMG_WIDTH / IMG_HEIGHT),
                1024 if IMG_HEIGHT > IMG_WIDTH else round(1024 * IMG_HEIGHT / IMG_WIDTH))
print("boxes: ")
print(boxes)
print("box_confidences: ")
print(box_confidences)
print("classes: ")
print(classes)
print("class_probs")
print(class_probs)
scores = []
for i in range(len(boxes)):
    score = box_confidences[i]*class_probs[i]
    scores.append(score)

cx, cy, w, h = prediction[0][:4] * np.array([IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH, IMG_HEIGHT])
x = round(cx - w / 2)

print("scores: ")
print(scores)

