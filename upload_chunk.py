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


    while True:
        time.sleep(0.5)
        if isFinished or len(frame_files) > 0:
            frame_dir = f'/home/mendel/VIP/frames/'
            frame_files = [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.mp4')]
            print(frame_files)
            print("upload started!")
            for frame_file in frame_files:
                try:
                    start_time = time.time()
                    original_name = os.path.basename(frame_file).split('.')
                    s3.upload_file(frame_file, 'fitchain-ai-videos', f'videos_input/{game_id}.{original_name[1]}', ExtraArgs = {'ACL':'public-read'})
                    print('uploaded!')
                    print(f"upload time: {time.time() - start_time}")
                    url = f"https://only-master-goldfish.ngrok-free.app/Inference/Run_Inference_In_Background/{game_id}"
                    response = requests.post(url)
#                    os.remove(frame_file)
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
    print("Started!")
    isRecording = True
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

    idex = 0
    filename = f"/home/mendel/VIP/frames/output_video_{index}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, resolution)

    # Record the video
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


    print('recording is done!')
    cv2.destroyAllWindows()

@sio.on('stop_recording')
def on_stop(data):
    global isRecording, isFinished
    print("Stopped!")
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
