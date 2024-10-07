import cv2
from PIL import Image

class Camera:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("Error: Could not open camera")

    def capture_image(self):
        ret, frame = self.camera.read()
        if ret:
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            print("Error: Could not capture frame with OpenCV")
            return None

    def release(self):
        self.camera.release()

    def isOpened(self):
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