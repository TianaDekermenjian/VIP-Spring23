import gi
import cv2
from time import sleep
from threading import Thread
import signal
import sys

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

Gst.init([])

def keyboard_interrupt_handler(signal, frame):
    print("KeyboardInterrupt: Exiting...")
    g_loop.quit()
    sys.exit(0)

signal.signal(signal.SIGINT, keyboard_interrupt_handler)

g_loop = GLib.MainLoop()
gl_thread = Thread(target=g_loop.run)
gl_thread.start()

# Updated GStreamer pipeline for RTSP streaming
pipeline_string = (
    "v4l2src device=/dev/video1 ! decodebin ! videoconvert ! "
    "x264enc ! rtph264pay name=pay0 pt=96"
)
pipeline = Gst.parse_launch(pipeline_string)

# Start the pipeline
pipeline.set_state(Gst.State.PLAYING)

# Create a new RTSP server
rtsp_server = Gst.ElementFactory.make("rtspsrc", "rtsp-source")
rtsp_server.set_property("location", "rtsp://127.0.0.1:8554/test")

# Link the elements and start streaming
pipeline.add(rtsp_server)
rtsp_server.link(pipeline.get_by_name("pay0"))

try:
    print("RTSP Streaming started. Press Ctrl+C to exit.")
    g_loop.run()
except KeyboardInterrupt:
    pass
finally:
    # Cleanly exit
    pipeline.set_state(Gst.State.NULL)
    sleep(1)  # Allow some time for the pipeline to clean up
    gl_thread.join()
    print("Exiting...")

