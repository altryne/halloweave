import cv2
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
                return self.camera.capture_array()
            except Exception as e:
                print(f"Error capturing image with PiCamera2: {e}")
                return None
        else:
            ret, frame = self.camera.read()
            if ret:
                return frame
            else:
                print("Error: Could not capture frame with OpenCV")
                return None

    def release(self):
        if not picamera_available:
            self.camera.release()

def get_camera():
    return Camera()

if __name__ == "__main__":
    camera = get_camera()
    image = camera.capture_image()
    
    if image is not None:
        cv2.imwrite('test_image.jpg', image)
        print("Image captured successfully")
        
        # Show the image
        cv2.imshow('Test Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Couldn't capture image")
    
    camera.release()