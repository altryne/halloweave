import time
import threading
import numpy as np
import pygame

try:
    import RPi.GPIO as GPIO
    ON_RASPBERRY_PI = True
except ImportError:
    print("RPi.GPIO not found. Running in simulation mode.")
    ON_RASPBERRY_PI = False

class SkeletonControl:
    def __init__(self):
        self.eyes_pin = 15
        self.mouth_pin = 23
        self.body_pin = 18

        self.eyes_lit = False
        self.body_moving = False
        self.mouth_speaking = False
        self.body_thread = None

        if ON_RASPBERRY_PI:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.eyes_pin, GPIO.OUT)
            GPIO.setup(self.mouth_pin, GPIO.OUT)
            GPIO.setup(self.body_pin, GPIO.OUT)

        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=1024)

    def eyes_on(self):
        if ON_RASPBERRY_PI:
            GPIO.output(self.eyes_pin, GPIO.HIGH)
        else:
            print("Eyes turned on")
        self.eyes_lit = True

    def eyes_off(self):
        if ON_RASPBERRY_PI:
            GPIO.output(self.eyes_pin, GPIO.LOW)
        else:
            print("Eyes turned off")
        self.eyes_lit = False

    def move_mouth(self, audio_buffer):
        """Move mouth based on audio buffer, distinguishing vowels, consonants, and silence."""
        # Debug print to show the received audio buffer
        print(f"Received audio buffer: length={len(audio_buffer)}, type={type(audio_buffer)}")
        print(f"First few bytes: {audio_buffer[:20]}")  # Print first 20 bytes as a sample
        sample_rate = 44100  # Hz
        chunk_duration = 0.1  # 100ms
        chunk_size = int(sample_rate * chunk_duration)
        total_chunks = len(audio_buffer) // 2 // chunk_size  # 2 bytes per sample for int16

        # Define vowel frequency range (e.g., 300Hz to 3000Hz)
        vowel_freq_min = 300
        vowel_freq_max = 3000

        # Define silence threshold
        silence_threshold = 500  # Adjust this value based on your audio characteristics

        audio_data = np.frombuffer(audio_buffer, dtype=np.int16)

        for i in range(total_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = audio_data[start:end]

            if len(chunk) == 0:
                continue

            # Check for silence
            if np.max(np.abs(chunk)) < silence_threshold:
                if ON_RASPBERRY_PI:
                    GPIO.output(self.mouth_pin, GPIO.LOW)
                else:
                    print("Mouth closed (silence)")
                time.sleep(chunk_duration)
                continue

            # Perform FFT on the chunk
            fft = np.fft.fft(chunk)
            freqs = np.fft.fftfreq(len(chunk), 1 / sample_rate)
            magnitudes = np.abs(fft)

            # Consider only positive frequencies
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitudes = magnitudes[:len(magnitudes)//2]

            # Calculate average magnitude in vowel range
            vowel_indices = np.where((positive_freqs >= vowel_freq_min) & (positive_freqs <= vowel_freq_max))[0]
            if len(vowel_indices) == 0:
                is_vowel = False
            else:
                avg_vowel_magnitude = np.mean(positive_magnitudes[vowel_indices])
                overall_avg_magnitude = np.mean(positive_magnitudes)
                
                # Threshold: if vowel magnitude is significantly higher than overall
                if avg_vowel_magnitude > 1.5 * overall_avg_magnitude:
                    is_vowel = True
                else:
                    is_vowel = False

            # Determine mouth movement duration
            if is_vowel:
                duration = 0.1  # 100ms for vowels
            else:
                duration = 0.02  # 20ms for consonants

            # Control the DC motor
            if duration > 0:
                if ON_RASPBERRY_PI:
                    GPIO.output(self.mouth_pin, GPIO.HIGH)
                    time.sleep(duration)
                    GPIO.output(self.mouth_pin, GPIO.LOW)
                else:
                    print(f"Mouth {'fully open' if is_vowel else 'slightly open'} for {duration:.3f} seconds")
            else:
                if ON_RASPBERRY_PI:
                    GPIO.output(self.mouth_pin, GPIO.LOW)
                else:
                    print("Mouth closed")

            # Maintain consistent timing
            remaining_time = chunk_duration - duration
            if remaining_time > 0:
                time.sleep(remaining_time)

    def start_body_movement(self):
        """Start the body movement."""
        if not self.body_moving:
            self.body_moving = True
            if ON_RASPBERRY_PI:
                GPIO.output(self.body_pin, GPIO.HIGH)
            else:
                print("Body movement started")
            self.body_thread = threading.Thread(target=self._body_movement_thread)
            self.body_thread.start()

    def stop_body_movement(self):
        """Stop the body movement."""
        if self.body_moving:
            self.body_moving = False
            if ON_RASPBERRY_PI:
                GPIO.output(self.body_pin, GPIO.LOW)
            else:
                print("Body movement stopped")
            if self.body_thread:
                self.body_thread.join()

    def _body_movement_thread(self):
        """Thread function for body movement."""
        while self.body_moving:
            time.sleep(0.1)  # Small delay to prevent busy-waiting

    def get_state(self):
        """Return the current state of the skeleton."""
        return {
            "eyes_lit": self.eyes_lit,
            "body_moving": self.body_moving,
            "mouth_speaking": self.mouth_speaking
        }

    def cleanup(self):
        self.stop_body_movement()
        if ON_RASPBERRY_PI:
            GPIO.cleanup()
        else:
            print("Cleanup performed")
        self.eyes_lit = False
        self.body_moving = False
        self.mouth_speaking = False

    def play_audio_and_move_mouth(self, audio_file):
        """Play audio and move mouth in a separate thread."""
        sound = pygame.mixer.Sound(audio_file)
        audio_array = pygame.sndarray.array(sound)
        
        def audio_thread():
            channel = sound.play()
            self.move_mouth(audio_array.tobytes())
            while channel.get_busy():
                pygame.time.wait(100)
        
        threading.Thread(target=audio_thread, daemon=True).start()

if __name__ == "__main__":
    skeleton = SkeletonControl()
    
    print("Booting sequence initiated...")
    print("Starting diagnostics skeleton control test...")
    
    # Start audio playback and mouth movement in a separate thread
    print("Testing audio playback and mouth movement")
    audio_file = "/home/altryne/halloween/sounds/booting_seq.wav"
    skeleton.play_audio_and_move_mouth(audio_file)
    
    # Flash eyes a couple of times while audio is playing
    for _ in range(2):
        print("Eyes on")
        skeleton.eyes_on()
        time.sleep(2)
        print("Eyes off")
        skeleton.eyes_off()
        time.sleep(0.5)
    
    # Body movement test
    print("Starting body movement")
    skeleton.start_body_movement()
    time.sleep(5)
    print("Stopping body movement")
    skeleton.stop_body_movement()
    
    # Wait for audio to finish (optional)
    time.sleep(2)  # Adjust this value based on the length of your audio file
    
    print("Skeleton control test completed.")
    # skeleton.cleanup()