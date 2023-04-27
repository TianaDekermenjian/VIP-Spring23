import os
import socketio
import time
import cv2
import boto3
import threading

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Create an S3 client
s3 = boto3.client('s3', aws_access_key_id=os.getenv("ACCESS_KEY_ID"), aws_secret_access_key=os.getenv("AWS_SECRET_KEY"))

sio = socketio.Client()
isRecording = False

async def upload_thread():
    while isRecording:
        frame_dir = f'/home/mendel/VIP/VIP/frames/'
        frame_files = [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.png')]

        for frame_file in frame_files:
            await s3.upload_file(frame_file, 'fitchain', f'coral_recordings/{os.path.basename(frame_file)}')
            os.remove(frame_file)

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
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    camera.set(cv2.CAP_PROP_FPS, fps)

    # Record the video
    start_time = time.time()
    index = 0
    while isRecording and time.time() - start_time < 5:
        ret, frame = camera.read()
        if not ret:
            break

        index = index +1
        cv2.imwrite(f'/home/mendel/VIP/VIP/frames/frame{index}.png', frame)

        cv2.waitKey(50)

    # Release the camera and destroy the window when all recordings are complete
    camera.release()
    cv2.destroyAllWindows()

@sio.on('stop_recording')
def on_stop():
    global isRecording
    print("Stopped")
    isRecording = False

sio.connect('http://ec2-52-91-118-179.compute-1.amazonaws.com:3001')

t = threading.Thread(target=upload_thread)

t.start()
t.join()

sio.wait()