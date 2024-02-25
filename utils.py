import cv2
import yaml
import numpy as np
import pycoral.utils.edgetpu as etpu
from pycoral.adapters import common
from nms import non_max_suppression

class YOLOv5s:
    def __init__(self, weights, labelmap, conf, iou):
        self.weights = weights
        self.labelmap = labelmap
        self.conf = conf
        self.iou = iou

        self.interpreter = etpu.make_interpreter(self.weights)
        self.interpreter.allocate_tensors()

        self.input_tensor_details = self.interpreter.get_input_details()
        self.input_scale = self.input_tensor_details[0]['quantization'][0]
        self.input_zero = self.input_tensor_details[0]['quantization'][1]
        self.input_size = common.input_size(self.interpreter)
        self.input_data_type = self.input_tensor_details[0]['dtype']

        self.output_tensor_details = self.interpreter.get_output_details()
        self.output_scale = self.output_tensor_details[0]['quantization'][0]
        self.output_zero = self.output_tensor_details[0]['quantization'][1]
        self.output_data_type = self.output_tensor_details[0]['dtype']

    def load_classes(self, label_file):
        with open(label_file, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)

        classes = cfg['names']

        return classes

    def preprocess_frame(self, frame):
        self.frame = cv2.imread(frame)

        self.frame_h, self.frame_w, c = self.frame.shape

        original_frame_size = self.frame.shape[:2]
        ratio = float(self.input_size[0]/max(original_frame_size))
        new_size = tuple([int(x*ratio) for x in original_frame_size])

        frame_resized = cv2.resize(self.frame, (new_size[1], new_size[0]))

        self.pad_w = self.input_size[0] - new_size[1]
        self.pad_h = self.input_size[0] - new_size[0]
        self.pad = (self.pad_w, self.pad_h)
        color = [100, 100, 100]

        frame_padded = cv2.copyMakeBorder(frame_resized, 0, self.pad_h, 0, self.pad_w, cv2.BORDER_CONSTANT, value=color)
        frame_padded = frame_padded.astype(np.float32)

        frame_normalized = frame_padded/255.0

        if frame_normalized.shape[0] == 3:
            frame_normalized = frame_normalized.transpose((1,2,0))

        self.input_frame = (frame_normalized/self.input_scale) + self.input_zero
        self.input_frame = self.input_frame[np.newaxis].astype(self.input_data_type)

        return self.input_frame

    def inference(self, input_frame):
        self.interpreter.set_tensor(self.input_tensor_details[0]['index'], input_frame)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_tensor_details[0]["index"])
        output = (output.astype(np.float32) - self.output_zero) * self.output_scale

        nms_result = non_max_suppression(output, self.conf, self.iou, None, False, 10000)

        return nms_result

    def postprocess(self, results):
        detections = results[0]

        ratio_w = self.frame_w/(self.input_size[0] - self.pad_w)
        ratio_h = self.frame_h/(self.input_size[1] - self.pad_h)

        scaled_coordinates = []

        if len(detections):
            for coordinates in detections[:,:4]:
                x1, y1, x2, y2 = coordinates

                x1_scaled = max(0, (x1*self.input_size[0]*ratio_w))
                y1_scaled = max(0, (y1*self.input_size[1]*ratio_h))
                x2_scaled = min(self.frame_w, (x2*self.input_size[0]*ratio_w))
                y2_scaled = min(self.frame_h, (y2*self.input_size[1]*ratio_h))

                scaled_coordinates.append((int(x1_scaled), int(y1_scaled), int(x2_scaled), int(y2_scaled)))

            detections[:,:4] = scaled_coordinates

        return detections
