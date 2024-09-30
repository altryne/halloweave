# motion_detector.py

import cv2
import time

def motion_detector(callback, timeout=5):
    cap = cv2.VideoCapture(0)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    last_triggered = time.time() - timeout

    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours and (time.time() - last_triggered) > timeout:
            last_triggered = time.time()
            # Motion detected, capture the image
            ret, frame = cap.read()
            image_path = 'captured.jpg'
            cv2.imwrite(image_path, frame)
            # Call the callback function
            callback(image_path)

        frame1 = frame2
        ret, frame2 = cap.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()