import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
else:
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('test_image.jpg', frame)
        print("Image captured successfully")
    else:
        print("Couldn't capture image")
cap.release()