import RPi.GPIO as GPIO
import time
import random

class HalloweenProp:
    def __init__(self, mouth_pin=4, eyes_pin=17):
        self.mouth_pin = mouth_pin
        self.eyes_pin = eyes_pin
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.mouth_pin, GPIO.OUT)
        GPIO.setup(self.eyes_pin, GPIO.OUT)
        
        # Ensure both relays are off initially
        GPIO.output(self.mouth_pin, GPIO.LOW)
        GPIO.output(self.eyes_pin, GPIO.LOW)

    def open_mouth(self):
        GPIO.output(self.mouth_pin, GPIO.HIGH)

    def close_mouth(self):
        GPIO.output(self.mouth_pin, GPIO.LOW)

    def light_eyes(self):
        GPIO.output(self.eyes_pin, GPIO.HIGH)

    def dim_eyes(self):
        GPIO.output(self.eyes_pin, GPIO.LOW)

    def cleanup(self):
        GPIO.cleanup()

def main():
    prop = HalloweenProp()
    try:
        while True:
            print("Starting speech pattern...")
            for _ in range(10):  # Emulate a sentence with 10 sounds
                if random.random() < 0.3:  # 30% chance for a vowel sound
                    print("Vowel sound...")
                    prop.open_mouth()
                    prop.light_eyes()
                    time.sleep(0.4)  # Longer opening for vowels
                    prop.close_mouth()
                else:
                    print("Consonant sound...")
                    prop.open_mouth()
                    prop.light_eyes()
                    time.sleep(0.1)  # Shorter opening for consonants
                    prop.close_mouth()
                
                time.sleep(random.uniform(0.05, 0.2))  # Random pause between sounds
            
            print("End of sentence, pausing...")
            prop.dim_eyes()
            time.sleep(1.5)  # Pause between sentences
    except KeyboardInterrupt:
        print("Program stopped by user")
    finally:
        prop.cleanup()

if __name__ == "__main__":
    main()