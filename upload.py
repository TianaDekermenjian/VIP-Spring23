import os
import socketio
import time
import cv2
import boto3
import threading
import asyncio

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Create an S3 client
s3 = boto3.client('s3', aws_access_key_id=os.getenv("ACCESS_KEY_ID"), aws_secret_access_key=os.getenv("AWS_SECRET_KEY"))

sio = socketio.Client()
isRecording = False

async def upload_thread():
    print('enter thread')

    frame_files = []

    while True:
        time.sleep(0.1)
        if isRecording or len(frame_files) > 0:
            frame_dir = f'/home/mendel/VIP/frames/'
            frame_files = [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.png')]
            print(frame_files)
            for frame_file in frame_files:
                await s3.upload_file(frame_file, 'fitchain', f'coral_recordings/{os.path.basename(frame_file)}')
                print('uploaded')
                os.remove(frame_file)

def in_between():
   asyncio.run(upload_thread())

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
        #print('it is recording')
        if not ret:
            break

        index = index +1
        cv2.imwrite(f'/home/mendel/VIP/frames/frame{index}.png', frame)

        cv2.waitKey(1)
    stop_time = time.time()
    duration = stop_time - start_time
    print(duration)

    # Release the camera and destroy the window when all recordings are complete
    camera.release()
    print('recording is done')
    cv2.destroyAllWindows()

@sio.on('stop_recording')
def on_stop():
    global isRecording
    print("Stopped")
    isRecording = False

# Testing app
sio.connect('http://ec2-52-91-118-179.compute-1.amazonaws.com:3001')
# Testing locally
#sio.connect('http://192.168.100.60:5000')
t = threading.Thread(target=in_between)

t.start()
t.join()

sio.wait()
