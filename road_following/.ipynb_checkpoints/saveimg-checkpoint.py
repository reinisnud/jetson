import os
import time
from jetbot import Camera, bgr8_to_jpeg

interval_seconds = 5
directory = 'images'
camera = Camera()

# count = 0

# while True:
file_name = os.path.join(directory, 'image_1.jpeg')
with open(file_name, 'wb') as f:
    f.write(bgr8_to_jpeg(camera.value))
    # count +=  1
    # time.sleep(interval_seconds)