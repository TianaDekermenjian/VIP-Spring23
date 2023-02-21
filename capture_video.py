import time
import cv2

# Set the camera resolution and frame rate
resolution = (1920, 1080) # 1920x1080 is the maximum resolution for 30fps
fps = 30

# The frame size of the camera with a 4:3 aspect ratio, 105° horizontal field of view,
# and resolution of 1920x1080 is 1920x1076

# Define the crop region:
# If you wish to keep the original video, set x and y to 0, w = 1920, and h = 1076
x, y, w, h = 480, 269, 960, 538

# Initialize the ArduCam USB camera
camera = cv2.VideoCapture(1)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"mp4v"))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
camera.set(cv2.CAP_PROP_FPS, fps)

# Display video
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

for i in range(2):  # Set number of videos needed
    # Construct the filename with a timestamp and save it to the desired directory
    filename = "C:/Users/Diana/Desktop/test/video_" + str(i) + "_" + time.strftime("%Y%m%d-%H%M%S") + ".mp4"

    # Initialize the video writer with the output filename and codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))

    # Record the video and display the frames in a window
    recording_duration = 10 # Change accordingly
    start_time = time.time()
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        cropped_frame = frame[y:y+h, x:x+w]
        cv2.imshow("Video", cropped_frame)
        writer.write(cropped_frame)
        if time.time() - start_time > recording_duration:
            break
        # Wait for a short time to display the frames in the window
        cv2.waitKey(1)

    # Release the video writer and display a message when the recording is complete
    writer.release()
    print("Video recording saved as:", filename)

# Release the camera and destroy the window when all recordings are complete
camera.release()
cv2.destroyAllWindows()
