import cv2
from PIL import Image
import numpy as np

picamera_available = False
try:
    from picamera2 import Picamera2
    picamera_available = True
except ImportError:
    print("PiCamera2 not available, falling back to OpenCV")

class Camera:
    def __init__(self):
        global picamera_available
        if picamera_available:
            try:
                self.camera = Picamera2()
                self.camera.configure(self.camera.create_preview_configuration())
                self.camera.start()
                print("PiCamera2 initialized successfully")
            except Exception as e:
                print(f"Error initializing PiCamera2: {e}")
                picamera_available = False
                self.camera = cv2.VideoCapture(0)
        else:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                print("Error: Could not open camera")

    def capture_image(self):
        if picamera_available:
            try:
                array = self.camera.capture_array()
                return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
            except Exception as e:
                print(f"Error capturing image with PiCamera2: {e}")
                return None
        else:
            ret, frame = self.camera.read()
            if ret:
                return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                print("Error: Could not capture frame with OpenCV")
                return None

    def release(self):
        if not picamera_available:
            self.camera.release()

    def isOpened(self):
        if picamera_available:
            return True  # PiCamera2 doesn't have an isOpened() method, so we assume it's always open
        else:
            return self.camera.isOpened()

def get_camera():
    return Camera()

if __name__ == "__main__":
    camera = get_camera()
    image = camera.capture_image()
    
    if image is not None:
        image.save('test_image.jpg')
        print("Image captured successfully")
        
        # Show the image
        image.show()
    else:
        print("Couldn't capture image")
    
    camera.release()