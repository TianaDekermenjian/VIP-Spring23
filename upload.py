import os
import socketio
import time
import cv2
import boto3
import threading
import requests

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Create an S3 client
s3 = boto3.client('s3', aws_access_key_id=os.getenv("ACCESS_KEY_ID"), aws_secret_access_key=os.getenv("AWS_SECRET_KEY"))

sio = socketio.Client()
isRecording = False
isFinished = False

game_id = 0

def upload_thread():
    global isFinished
    print('enter thread')

    frame_files = []

    start_time = time.time()

    while True:
        time.sleep(0.1)
        if isFinished or len(frame_files) > 0:
            print("upload started")
            frame_dir = f'/home/mendel/VIP/frames/'
            frame_files = [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.mp4')]
            print(frame_files)
            for frame_file in frame_files:
                try:
                    original_name = os.path.basename(frame_file).split('.')
                    s3.upload_file(frame_file, 'fitchain-ai-videos', f'videos_input/{game_id}.{original_name[1]}', ExtraArgs = {'ACL':'public-read'})
                    print('uploaded')
                    print(f"upload time: {time.time() - start_time}")
                    url = f"https://polished-remotely-troll.ngrok-free.app/Inference/Run_Inference_In_Background/{game_id}"
                    response = requests.post(url)
                    os.remove(frame_file)
                    print(response)
                except Exception as e:
                    print(e)
            isFinished = False

@sio.on('connect')
def on_connect():
    print('Connected!')

@sio.on('start_recording')
def on_start(data):
    global isRecording, game_id
    game_id = data['data']
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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'/home/mendel/VIP/frames/output.mp4', fourcc, fps, resolution)

    # Record the video
    while isRecording:
        ret, frame = camera.read()
        if not ret:
            break

        out.write(frame)

        cv2.waitKey(1)

    # Release the camera and destroy the window when all recordings are complete
    camera.release()
    out.release()

    print('recording is done')
    cv2.destroyAllWindows()

@sio.on('stop_recording')
def on_stop(data):
    global isRecording, isFinished
    print("Stopped")
    isRecording = False
    isFinished = True

# Testing app
sio.connect('http://ec2-16-170-232-235.eu-north-1.compute.amazonaws.com:3001')
# Testing locally
#sio.connect('http://192.168.100.60:5000')
t = threading.Thread(target=upload_thread)

t.start()
t.join()

sio.wait()
