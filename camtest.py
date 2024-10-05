from camera import get_camera
import cv2

def main():
    camera = get_camera()
    image = camera.capture_image()
    
    if image is not None:
        cv2.imwrite('test_image.jpg', image)
        print("Image captured successfully")
    else:
        print("Couldn't capture image")
    
    camera.release()

if __name__ == "__main__":
    main()