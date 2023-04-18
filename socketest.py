import socketio
import time
import cv2
sio = socketio.Client()
isRecording = False

@sio.on('connect')
def on_connect():
    print('Connected!')

@sio.on('start_recording')
def on_start():
    global isRecording
    print("Started")
    isRecording = True
    # Set the camera resolution and frame rate
    resolution = (640, 480)
    fps = 20

    # Initialize the ArduCam USB camera
    camera = cv2.VideoCapture(1)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"mp4v"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    camera.set(cv2.CAP_PROP_FPS, fps)

    # Construct the filename with a timestamp and save it to the desired directory
    filename = "~/VIP/video.mp4"

    # Initialize the video writer with the output filename and codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filename, fourcc, fps, (640, 480))

    # Record the video and display the frames in a window
    start_time = time.time()
    while isRecording:
        ret, frame = camera.read()
        if not ret:
            break
        writer.write(frame)
        # Wait for a short time to display the frames in the window
        cv2.waitKey(1)

    # Release the video writer and display a message when the recording is complete
    writer.release()
    print("Video recording saved as:", filename)

    # Release the camera and destroy the window when all recordings are complete
    camera.release()
    cv2.destroyAllWindows()

@sio.on('stop_recording')
def on_stop():
    global isRecording
    print("Stopped")
    isRecording = False

sio.connect('http://ec2-52-91-118-179.compute-1.amazonaws.com:3001')
sio.wait()