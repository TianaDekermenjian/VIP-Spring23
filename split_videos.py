import os
import cv2

# Set the path to the folder containing the videos
video_folder = r"C:\Users\Diana\Desktop\test"

for filename in os.listdir(video_folder):
    if filename.endswith(".mp4"):
        # Load the video
        video_path = os.path.join(video_folder, filename)
        video = cv2.VideoCapture(video_path)

        # To check if video is being opened
        # print(video.isOpened())

        # To check frame count
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Frame count:", frame_count)

        # Initialize frame count
        count = 0

        # Loop through each frame in the video
        while True:
            # Read a frame
            ret, frame = video.read()
            # Check if frame is being read
            # print("Frame read:", ret)

            # If the frame was not successfully read, exit the loop
            if not ret:
                break

            # Save the frame as an image file
            frame_path = os.path.join(video_folder, "frames", f"{filename}_frame{count}.jpg")
            cv2.imwrite(frame_path, frame)

            # Check if frames are being saved
            # if os.path.exists(frame_path):
            #     print(f"Saved frame {count} to {frame_path}")
            # else:
            #     print(f"Error: Failed to save frame {count} to {frame_path}")

            # Increment the frame count
            count += 1

        # Release the video capture object
        video.release()

# Close all windows
cv2.destroyAllWindows()
