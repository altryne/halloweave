import threading
import time

try:
    import RPi.GPIO as GPIO
    ON_RASPBERRY_PI = True
except ImportError:
    print("RPi.GPIO not found. Running in simulation mode.")
    ON_RASPBERRY_PI = False

class SkeletonControl:
    def __init__(self):
        self.eyes_pin = 17
        self.mouth_pin = 4
        self.body_pin = 27

        if ON_RASPBERRY_PI:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.eyes_pin, GPIO.OUT)
            GPIO.setup(self.mouth_pin, GPIO.OUT)
            GPIO.setup(self.body_pin, GPIO.OUT)
        
        self.mouth_thread = None
        self.mouth_moving = False
        self.body_thread = None
        self.body_moving = False

    def eyes_on(self):
        if ON_RASPBERRY_PI:
            GPIO.output(self.eyes_pin, GPIO.HIGH)
        else:
            print("Eyes turned on")

    def eyes_off(self):
        if ON_RASPBERRY_PI:
            GPIO.output(self.eyes_pin, GPIO.LOW)
        else:
            print("Eyes turned off")

    def start_mouth_movement(self):
        self.mouth_moving = True
        self.mouth_thread = threading.Thread(target=self._move_mouth)
        self.mouth_thread.start()

    def stop_mouth_movement(self):
        self.mouth_moving = False
        if self.mouth_thread:
            self.mouth_thread.join()

    def _move_mouth(self):
        """Default mouth movement."""
        while self.mouth_moving:
            self.dynamic_mouth_movement(0.1)  # Default interval for basic movement

    def dynamic_mouth_movement(self, duration):
        """Control the mouth with dynamic timing based on audio amplitude."""
        if ON_RASPBERRY_PI:
            GPIO.output(self.mouth_pin, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(self.mouth_pin, GPIO.LOW)
        else:
            print(f"Mouth moving for {duration} seconds")
            time.sleep(duration)

    def start_body_movement(self):
        self.body_moving = True
        self.body_thread = threading.Thread(target=self._move_body)
        self.body_thread.start()

    def stop_body_movement(self):
        self.body_moving = False
        if self.body_thread:
            self.body_thread.join()

    def _move_body(self):
        while self.body_moving:
            if ON_RASPBERRY_PI:
                GPIO.output(self.body_pin, GPIO.HIGH)
                time.sleep(0.1)
                GPIO.output(self.body_pin, GPIO.LOW)
                time.sleep(0.1)
            else:
                print("Body moving")
                time.sleep(0.2)

    def cleanup(self):
        if ON_RASPBERRY_PI:
            GPIO.cleanup()
        else:
            print("Cleanup performed")