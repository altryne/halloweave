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
        """Move mouth based on audio buffer and word timings."""
        # print(f"Received audio buffer: length={len(audio_buffer)}, type={type(audio_buffer)}")
        # print(f"First few bytes: {audio_buffer[:20]}")
        
        sample_rate = 22050  # Hz (updated to match the new rate)
        chunk_duration = 0.02  # 50ms chunks for more granular control
        chunk_size = int(sample_rate * chunk_duration)
        total_chunks = len(audio_buffer) // 2 // chunk_size  # 2 bytes per sample for int16

        silence_threshold = 300  # Adjust this value based on your audio characteristics

        audio_data = np.frombuffer(audio_buffer, dtype=np.int16)

        current_time = 0
        last_word_end = 0
        for i in range(total_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = audio_data[start:end]

            if len(chunk) == 0:
                continue

            # Check for silence
            if np.max(np.abs(chunk)) < silence_threshold:
                self._set_mouth_state(False)
                time.sleep(chunk_duration)
                current_time += chunk_duration
                continue

            # If we have word timings, use them for more accurate mouth movement
            if word_timings:
                is_speaking = False
                for word, word_start, word_end in zip(word_timings['words'], word_timings['start'], word_timings['end']):
                    if word_start <= current_time < word_end:
                        is_speaking = True
                        last_word_end = word_end
                        break
                
                # Close mouth between words
                if current_time > last_word_end + 0.1:  # Add a small delay after each word
                    is_speaking = False
                
                self._set_mouth_state(is_speaking)
            else:
                # Fallback to simple amplitude-based mouth movement
                self._set_mouth_state(True)

            time.sleep(chunk_duration)
            current_time += chunk_duration

        # Ensure mouth is closed at the end
        self._set_mouth_state(False)

    def _set_mouth_state(self, is_open):
        if ON_RASPBERRY_PI:
            GPIO.output(self.mouth_pin, GPIO.HIGH if is_open else GPIO.LOW)
        else:
            print(f"Mouth {'open' if is_open else 'closed'}")

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
