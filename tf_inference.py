TEST_FILE = r'data\OurDatasets\HardwareDatasetOne\images\test\HardwareDatasetOne_100248_jpg.rf.dba4a6aaf21219531fafd33cd1748d29.jpg'
TF_LITE_MODEL = r'runs\train\yolo_cuda_test\weights\best-fp16.tflite'
LABEL_MAP = r'data\labelmap.txt'
BOX_THRESHOLD = 0.5
CLASS_THRESHOLD = 0.5
LABEL_SIZE = 0.5

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(TF_LITE_MODEL, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

_, height, width, _ = interpreter.get_input_details()[0]['shape']

with open(LABEL_MAP, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

img = cv2.imread(TEST_FILE, cv2.IMREAD_COLOR)
IMG_HEIGHT, IMG_WIDTH = img.shape[:2]

pad = round(abs(IMG_WIDTH - IMG_HEIGHT) / 2)
x_pad = pad if IMG_HEIGHT > IMG_WIDTH else 0
y_pad = pad if IMG_WIDTH > IMG_HEIGHT else 0
img_padded = cv2.copyMakeBorder(img, top=y_pad, bottom=y_pad, left=x_pad, right=x_pad,
                                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
IMG_HEIGHT, IMG_WIDTH = img_padded.shape[:2]

img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_AREA)
input_data = np.expand_dims(img_resized / 255, axis=0).astype('float32')

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

outputs = interpreter.get_tensor(output_details[0]['index'])[0]

boxes = []
box_confidences = []
classes = []
class_probs = []

max_confidence = 0
best_box = None
best_class = None
best_class_prob = 0

for output in outputs:
    box_confidence = output[4]
    if box_confidence < BOX_THRESHOLD:
        continue

    class_ = output[5:].argmax(axis=0)
    class_prob = output[5:][class_]

    if class_prob < CLASS_THRESHOLD:
        continue

    if box_confidence * class_prob > max_confidence:
        max_confidence = box_confidence * class_prob
        best_box = output[:4] * np.array([IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH, IMG_HEIGHT])
        best_class = class_
        best_class_prob = class_prob

if best_box is not None:
    x, y, w, h = (best_box).astype(int)
    class_name = labels[best_class]
    score = max_confidence
    color = [int(c) for c in colors[best_class]]
    text_color = (255, 255, 255) if sum(color) < 144 * 3 else (0, 0, 0)

    cv2.rectangle(img_padded, (x, y), (x + w, y + h), color, 2)

    label = f'{class_name}: {score * 100:.2f}%'
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SIZE, 2)
    cv2.rectangle(img_padded,
                  (x, y + baseLine), (x + labelSize[0], y - baseLine - labelSize[1]),
                  color, cv2.FILLED)
    cv2.putText(img_padded, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, LABEL_SIZE, text_color, 1)

img_show = img_padded[y_pad: IMG_HEIGHT - y_pad, x_pad: IMG_WIDTH - x_pad]
cv2.namedWindow('Object detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object detection',
                 1024 if IMG_WIDTH > IMG_HEIGHT else round(1024 * IMG_WIDTH / IMG_HEIGHT),
                 1024 if IMG_HEIGHT > IMG_WIDTH else round(1024 * IMG_HEIGHT / IMG_WIDTH))
cv2.imshow('Object detection', img_show)
cv2.imwrite('./result.jpg', img_show)
cv2.imwrite('./result_yolo.jpg', img_show)
cv2.waitKey(0)
cv2.destroyAllWindows()