import time
import cv2
import numpy as np

def start_recording(double time):
    # Set the camera resolution and frame rate
    resolution = (640, 480)
    fps = 20

    measure1 = []
    measure2 = []

    # Initialize the ArduCam USB camera
    camera = cv2.VideoCapture(1)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"mp4v"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    camera.set(cv2.CAP_PROP_FPS, fps)

    # Construct the filename with a timestamp and save it to the desired directory


    # Initialize the video writer with the output filename and codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filename, fourcc, fps)

    # Record the video and display the frames in a window
    start_time = time.time()
    start_time2 = time.time()
    index = 0
    while time.time() - start_time < time:
        filename = f"./Video/video_{index}.mp4"
        if time.time() - start_time2 >= 10:
            # Release the video writer and display a message when the recording is complete
            writer.release()
            index += 1
            print(f"Video recording saved as:, {filename}")
            measure1.append(time.time())

            # Initialize the ArduCam USB camera
            camera = cv2.VideoCapture(1)
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"mp4v"))
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            camera.set(cv2.CAP_PROP_FPS, fps)

            # Construct the filename with a timestamp and save it to the desired directory

            # Initialize the video writer with the output filename and codec
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(filename, fourcc, fps)

            measure2.append(time.time())

        ret, frame = camera.read()
        if not ret:
            break
        writer.write(frame)
        # Wait for a short time to display the frames in the window
        cv2.waitKey(1)
    else:
        writer.release()
        filename = f"./Video/video_{index}.mp4"
        index += 1
        print(f"Video recording saved as:, {filename}")


    print ("time between epsilones:", np.subtract(measure2,measure1))

    # Release the camera and destroy the window when all recordings are complete
    camera.release()
    cv2.destroyAllWindows()

