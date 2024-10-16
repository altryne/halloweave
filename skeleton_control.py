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
        self.eyes_body_pin = 18  # green on the pi - pink out of the ULN2803
        self.mouth_pin = 4  # red on the pi - orange out of the ULN2803

        self.eyes_lit = False
        self.body_moving = False
        self.mouth_speaking = False
        self.eyes_body_pwm = None
        self.mouth_open = False

        if ON_RASPBERRY_PI:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.eyes_body_pin, GPIO.OUT)
            GPIO.setup(self.mouth_pin, GPIO.OUT)
            
            try:
                self.eyes_body_pwm = GPIO.PWM(self.eyes_body_pin, 1000)  # 1000 Hz frequency
                self.eyes_body_pwm.start(0)  # Start with 0% duty cycle
            except Exception as e:
                print(f"Failed to initialize PWM for eyes: {e}")
                self.eyes_body_pwm = None

        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=1024)

        # Easily adjustable parameters for mouth movement
        self.silence_threshold = 0.1
        self.vowel_threshold = 0.35
        self.segment_duration = 0.02  # 20ms segments
        self.mouth_update_delay = 0.01  # 10ms delay between mouth updates
        self.current_mouth_state = 0

    def eyes_on(self):
        if ON_RASPBERRY_PI and self.eyes_body_pwm:
            try:
                self.eyes_body_pwm.ChangeDutyCycle(90)  # 90% brightness for eyes
            except Exception as e:
                print(f"Error turning eyes on: {e}")
        else:
            print("Eyes turned on (simulation mode)")
        self.eyes_lit = True

    def eyes_off(self):
        if ON_RASPBERRY_PI and self.eyes_body_pwm:
            if not self.body_moving:
                try:
                    self.eyes_body_pwm.ChangeDutyCycle(0)  # Off
                except Exception as e:
                    print(f"Error turning eyes off: {e}")
        else:
            print("Eyes turned off (simulation mode)")
        self.eyes_lit = False

    def move_mouth(self, audio_buffer, word_timings=None):
        """Move mouth based on audio buffer, analyzing smaller segments within the chunk."""
        sample_rate = 44100  # Hz
        audio_data = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32)
        audio_data /= np.max(np.abs(audio_data))  # Normalize

        segment_size = int(self.segment_duration * sample_rate)
        num_segments = 2

        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size
            segment = audio_data[start:end]
            
            segment_amplitude = np.mean(np.abs(segment))
            
            if segment_amplitude < self.silence_threshold:
                self._set_mouth_state(0)  # Closed
                time.sleep(self.mouth_update_delay)
            elif segment_amplitude > self.vowel_threshold:
                self._set_mouth_state(1)
                time.sleep(self.mouth_update_delay * 5)
            else:
                openness = (segment_amplitude - self.silence_threshold) / (self.vowel_threshold - self.silence_threshold)
                self._set_mouth_state(openness)
                time.sleep(self.mouth_update_delay)
            
            print(f"Segment {i+1}/{num_segments}, Amplitude: {segment_amplitude:.4f}, Mouth openness: {self._get_mouth_state():.2f}")
            
            

        # Ensure the mouth is closed after processing all segments
        self._set_mouth_state(0)

    def stop_mouth_movement(self):
        self.mouth_speaking = False
        self._set_mouth_state(0)  # Ensure the mouth is closed
        
    def _set_mouth_state(self, openness):
        """Set the mouth state with a value between 0 (closed) and 1 (fully open)."""
        self.current_mouth_state = max(0, min(1, openness))  # Ensure openness is between 0 and 1
        if ON_RASPBERRY_PI:
            # Assuming you're using PWM for more granular control
            duty_cycle = int(self.current_mouth_state * 100)
            if openness > self.vowel_threshold:
                GPIO.output(self.mouth_pin, GPIO.HIGH)
            elif openness < self.silence_threshold:
                try:
                    GPIO.output(self.mouth_pin, GPIO.LOW)
                except Exception as e:
                    print(f"Error setting mouth state: {e}")
            else:
                try:
                    GPIO.output(self.mouth_pin, duty_cycle)
                except Exception as e:
                    print(f"Error setting mouth state: {e}")
        else:
            print(f"Mouth openness: {self.current_mouth_state:.2f}")
        
        # Update the mouth_open flag
        self.mouth_open = self.current_mouth_state > 0

    def _get_mouth_state(self):
        """Get the current mouth state."""
        return self.current_mouth_state

    def start_body_movement(self):
        """Start the body movement."""
        if not self.body_moving:
            self.body_moving = True
            if ON_RASPBERRY_PI and self.eyes_body_pwm:
                try:
                    self.eyes_body_pwm.ChangeDutyCycle(100)  # Full power for body movement
                except Exception as e:
                    print(f"Error starting body movement: {e}")
            else:
                print("Body movement started (simulation mode)")

    def stop_body_movement(self):
        """Stop the body movement."""
        if self.body_moving:
            self.body_moving = False
            if ON_RASPBERRY_PI and self.eyes_body_pwm:
                try:
                    if self.eyes_lit:
                        self.eyes_body_pwm.ChangeDutyCycle(90)  # Back to 90% for eyes
                    else:
                        self.eyes_body_pwm.ChangeDutyCycle(0)  # Off if eyes were off
                except Exception as e:
                    print(f"Error stopping body movement: {e}")
            else:
                print("Body movement stopped (simulation mode)")

    def get_state(self):
        """Return the current state of the skeleton."""
        return {
            "eyes_lit": self.eyes_lit,
            "body_moving": self.body_moving,
            "mouth_speaking": self.mouth_speaking
        }

    def cleanup(self):
        self.stop_body_movement()
        self.stop_mouth_movement()  # Ensure the mouth is closed during cleanup
        if ON_RASPBERRY_PI:
            GPIO.cleanup()
        else:
            print("Cleanup performed (simulation mode)")
        self.eyes_lit = False
        self.body_moving = False
        self.mouth_speaking = False

    def play_audio_and_move_mouth(self, audio_file, word_timings=None):
        """Play audio and move mouth in a separate thread, simulating streaming."""
        sound = pygame.mixer.Sound(audio_file)
        audio_array = pygame.sndarray.array(sound)
        # multiply the volume by 2
        audio_array = (audio_array * 2).astype(np.int16)
        
        def audio_thread():
            channel = sound.play()
            
            # Calculate segment size and number of segments
            sample_rate = 44100  # Hz
            segment_size = int(self.segment_duration * sample_rate)
            total_segments = len(audio_array) // segment_size
            chunks_of_5 = total_segments // 5
            
            for chunk in range(chunks_of_5):
                start = chunk * 5 * segment_size
                end = start + 5 * segment_size
                chunk_data = audio_array[start:end]
                
                # Process this chunk
                self.move_mouth(chunk_data.tobytes(), word_timings)
                
                # Simulate delay between chunks
                time.sleep(self.segment_duration * 5)
            
            # Process any remaining data
            remaining_data = audio_array[chunks_of_5 * 5 * segment_size:]
            if len(remaining_data) > 0:
                self.move_mouth(remaining_data.tobytes(), word_timings)
            
            while channel.get_busy():
                pygame.time.wait(100)
        
        threading.Thread(target=audio_thread, daemon=True).start()

if __name__ == "__main__":
    skeleton = SkeletonControl()

    print("Booting sequence initiated...")
    print("Starting diagnostics skeleton control test...")
    
    print("Starting audio playback and mouth movement")
    audio_file = "/home/altryne/halloween/sounds/booting_seq.wav"
    skeleton.play_audio_and_move_mouth(audio_file)
    
    print("Turning eyes on")
    skeleton.eyes_on()
    time.sleep(1)
    
    print("Turning eyes off")
    skeleton.eyes_off()
    time.sleep(1)
    
    print("Turning eyes on again for 3 seconds")
    skeleton.eyes_on()
    time.sleep(3)
    
    print("Starting body movement for 3 seconds")
    skeleton.start_body_movement()
    time.sleep(3)
    skeleton.stop_body_movement()
    
    print("Waiting for audio to finish...")
    # You might want to add a way to check if the audio is still playing
    # For now, we'll just wait for a fixed amount of time
    time.sleep(5)
    
    print("Turning everything off")
    skeleton.eyes_off()
    
    print("Skeleton control test completed.")
    skeleton.cleanup()
