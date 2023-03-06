from tflite_runtime.interpreter import Interpreter
import cv2
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the label file
labels_path = “./models/VIP-Spring23/sunnylabels.txt”
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the model
model_path = "./models/VIP-Spring23/sunny_edgetpu.tflite”
interpreter = Interpreter(model_path=model_path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set camera resolution to ???x???, FPS to ???
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, ???)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ???)
cap.set(cv2.CAP_PROP_FPS, ???)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Resize the input frame to the size expected by the model
    input_shape = input_details[0]['shape'][1:3]
    frame = cv2.resize(frame, input_shape)

    # Preprocess the input frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype('float32') / 255.0
    frame = (frame - input_details[0]['quantization'][0]) / input_details[0]['quantization'][1]

    # Set the input tensor
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    # Run inference
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output_tensor = interpreter.get_tensor(output_details[0]['index'])
    num_detections = int(output_tensor[0])

    # Postprocess the output to get the bounding boxes and class labels
    for i in range(num_detections):
        class_id = int(output_tensor[1][i])
        score = float(output_tensor[2][i])
        bbox = output_tensor[3][i]
        xmin = int(bbox[0] * frame.shape[1])
        ymin = int(bbox[1] * frame.shape[0])
        xmax = int(bbox[2] * frame.shape[1])
        ymax = int(bbox[3] * frame.shape[0])
        label = labels[class_id]

        # Draw the bounding box and label on the frame
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}: {score:.2f}', (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the output frame on the screen
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()
