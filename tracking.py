import os
import time
import pathlib
import cv2
import numpy as np
import traceback
from periphery import PWM
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.adapters import detect
from PID import PID

controller = PID(0.01, 0, 0)

# Open PWM pin
pwm = PWM(1, 0)

pwm.frequency = 50
pwm.duty_cycle = 0.05

pwm.enable()

# Set camera resolution and FPS accordingly
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 20)

# Specify the TensorFlow model and labels
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, './models (edgetpu)/sunny_edgetpu.tflite')
# label_file = os.path.join(script_dir, 'REPLACE')

# Initialize the TF interpreter
interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()
output_details = interpreter.get_output_details()

try:
    while True:
        # Measure inference time
        st = time.perf_counter_ns()
        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        size = common.input_size(interpreter)
        image = cv2.resize(frame, size)

        # Run an inference
        common.set_input(interpreter, image)
        interpreter.invoke()
        classes = classify.get_classes(interpreter, top_k=1)

        objs = detect.get_objects(interpreter, 0.4, [1, 1])

        # Print the result
        # labels = dataset.read_label_file(label_file)
        # for c in classes:
        #     print('%s: %.5f' % (labels.get(c.id, c.id), c.score))

        output_tensor = interpreter.get_tensor(output_details[0]['index'])
        # print(output_tensor.shape, output_tensor)
        num_detections = len(output_tensor[0])

        for obj in objs:
            print("obj")
            # print(labels.get(obj.id, obj.id))
            print('  id:    ', obj.id)
            print('  score: ', obj.score)
            print('  bbox:  ', obj.bbox)

            cv2.rectangle(frame, (obj.bbox.xmin, obj.bbox.xmax), (obj.bbox.ymin, obj.bbox.ymax), (0, 255, 0), 2)

        print("Inference Time: ", (time.perf_counter_ns() - st) * 1e-6)

        if len(objs) >= 1:
            # Get center of bbox and center of frame
            center_frame = frame.shape[1] / 2
            center_obj = (objs[0].bbox.xmin + objs[0].bbox.xmax) / 2

            # Get the offset and correction from controller
            error = center_obj - center_frame
            corr = controller(error)

            # Update PWM
            pwm.duty_cycle = np.clip(pwm.duty_cycle + corr, 0.9, 0.95)

            print(corr, error, pwm.duty_cycle)

        # Display the output frame on the screen
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break
except KeyboardInterrupt:
    pass
# except Exception as e:
#     print("Error", e)
#     traceback.print_exc()

pwm.close()

# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()

print('ciao ciao')
