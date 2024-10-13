import time
import threading
import numpy as np
import pygame
import wave
import pyaudio

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

        # Easily adjustable parameters for mouth movement
        self.silence_threshold = 1000
        self.vowel_threshold = 8000
        self.vowel_duration = 0.06
        self.consonant_duration = 0.03
        self.consonant_openness = 0.5

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

    def move_mouth(self, audio_buffer, word_timings=None):
        """Move mouth based on audio buffer, distinguishing between vowels, consonants, and silence."""
        sample_rate = 22050  # Hz
        chunk_duration = len(audio_buffer) / (2 * sample_rate)  # 2 bytes per sample for int16
        # print(f"Chunk duration: {chunk_duration} seconds")
        audio_data = np.frombuffer(audio_buffer, dtype=np.int16)

        amplitude = np.max(np.abs(audio_data))
        print(f"Amplitude: {amplitude}")

        if amplitude < self.silence_threshold:
            # Silence
            self._set_mouth_state(0)
        elif amplitude > self.vowel_threshold:
            # Vowel (fully open)
            self._set_mouth_state(self.vowel_duration)
        else:
            # Consonant (partially open)
            self._set_mouth_state(self.consonant_duration)

        # # Sleep for a short duration to allow for visual feedback
        # time.sleep(min(chunk_duration, 0.05))  # Cap at 50ms to prevent long delays

    def _set_mouth_state(self, openness):
        """Set the mouth state with a value between 0 (closed) and 1 (fully open)."""
        if ON_RASPBERRY_PI:
            # Assuming you're using PWM for more granular control
            duty_cycle = int(openness * 500)
            GPIO.output(self.mouth_pin, duty_cycle)
        else:
            print(f"Mouth openness: {openness:.2f}")

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

    def play_audio_and_move_mouth(self, audio_file, word_timings=None):
        """Play audio and move mouth in a separate thread."""
        sound = pygame.mixer.Sound(audio_file)
        audio_array = pygame.sndarray.array(sound)
        
        def audio_thread():
            channel = sound.play()
            self.move_mouth(audio_array.tobytes(), word_timings)
            while channel.get_busy():
                pygame.time.wait(100)
        
        threading.Thread(target=audio_thread, daemon=True).start()

if __name__ == "__main__":
    skeleton = SkeletonControl()

    
    print("Booting sequence initiated...")
    print("Starting diagnostics skeleton control test...")
    
    print("Testing audio playback and mouth movement")
    audio_file = "/home/altryne/halloween/sounds/booting_seq.wav"
    
    CHUNK = 4608  # Reduced chunk size for more frequent updates
    wf = wave.open(audio_file, 'rb')
    p = pyaudio.PyAudio()

    # Force the stream to use 22050 Hz
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=22050,  # Force 22050 Hz
                    output=True,
                    frames_per_buffer=CHUNK)

    # Resample the audio if necessary
    original_rate = wf.getframerate()
    resampling_factor = 22050 / original_rate

    data = wf.readframes(int(CHUNK * resampling_factor))

    while len(data) > 0:
        if original_rate != 22050:
            # Simple linear interpolation for resampling
            audio_array = np.frombuffer(data, dtype=np.int16)
            resampled_array = np.interp(
                np.linspace(0, len(audio_array), int(len(audio_array) * resampling_factor)),
                np.arange(len(audio_array)),
                audio_array
            ).astype(np.int16)
            resampled_data = resampled_array.tobytes()
        else:
            resampled_data = data

        stream.write(resampled_data)
        skeleton.move_mouth(resampled_data)
        data = wf.readframes(int(CHUNK * resampling_factor))

    stream.stop_stream()
    stream.close()
    p.terminate()
    
    print("Skeleton control test completed.")
    skeleton.cleanup()
