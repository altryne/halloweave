import os
import wave
import time
import numpy as np
import pyaudio

# Generate a beep sound (frequency: 1000 Hz, duration: 500 ms)
def generate_beep():
    sample_rate = 44100
    frequency = 1000
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    beep = 0.5 * np.sin(2 * np.pi * frequency * t)
    beep = np.int16(beep * 32767)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, output=True)
    stream.write(beep.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

# Record audio for 5 seconds
def record_audio(filename="recorded_audio.wav", duration=5, sample_rate=44100):
    print("Recording...")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=1024)
    frames = []
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
    print("Recording finished")

# Play back the recorded audio
def play_audio(filename="recorded_audio.wav"):
    print("Playing back the recorded audio...")
    p = pyaudio.PyAudio()
    with wave.open(filename, 'rb') as wf:
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)
        stream.stop_stream()
        stream.close()
    p.terminate()

if __name__ == "__main__":
    generate_beep()
    time.sleep(0.5)  # Short delay to ensure the beep finishes playing
    record_audio()
    play_audio()