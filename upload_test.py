import time
import cv2
import numpy as np

def start_recording(times):
    # Set the camera resolution and frame rate
    resolution = (640, 480)
    fps = 20

    measure1 = []
    measure2 = []

    # Initialize the ArduCam USB camera
    camera = cv2.VideoCapture(1)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    camera.set(cv2.CAP_PROP_FPS, fps)

    start_time = time.time()
    start_time2 = time.time()

    # Construct the filename with a timestamp and save it to the desired directory

    index = 0
    filename = f"/home/mendel/VIP/VIP-Spring23/video_{index}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filename, fourcc, fps, resolution)

    while time.time() - start_time < times:
        ret, frame = camera.read()
        if not ret:
            break
        writer.write(frame)

        if time.time() - start_time2 >= 10:
            # Release the video writer and display a message when the recording is complete
            writer.release()
            index += 1
            print(f"Video recording saved as:, {filename}")
            measure1.append(time.time())

            filename = f"/home/mendel/VIP/VIP-Spring23/video_{index}.mp4"
            writer = cv2.VideoWriter(filename, fourcc, fps, resolution)

            measure2.append(time.time())
            start_time2 = time.time()

        cv2.waitKey(1)

    else:
        writer.release()
        filename = f"/home/mendel/VIP/VIP-Spring23/video_{index}.mp4"
        index += 1
        print(f"Video recording saved as:, {filename}")

    print("time between epsilones:", np.subtract(measure2, measure1))

    # Release the camera and destroy the window when all recordings are complete
    camera.release()
    cv2.destroyAllWindows()

start_recording(30)