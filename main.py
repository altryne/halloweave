import os
import asyncio
import time
import struct
import threading
from contextlib import asynccontextmanager
from threading import Lock, Thread
from queue import Queue
from io import BytesIO

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

import cv2
import pyaudio
from PIL import Image
import numpy as np

from cartesia import AsyncCartesia
import weave

from dotenv import load_dotenv

# Import the chat model functions
from gemini import gemini_chat
from openai_client import openai_chat
from openrouter import openrouter_chat

# Import Porcupine and PvRecorder
import pvporcupine
from pvrecorder import PvRecorder

load_dotenv()

# Global variables for camera and motion detection
camera = None
camera_lock = Lock()
frame_queue = Queue(maxsize=10)  # Buffer for frames
motion_detected = False
last_motion_time = 0
audio_playing = False

# Choose the chat model to use
CHAT_MODEL = "gemini"  # Options: "gemini", "openai", "openrouter"

# Porcupine wake word detection
porcupine = None
recorder = None

# Flag to enable/disable motion detection
ENABLE_MOTION_DETECTION = False

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global camera, porcupine, recorder
    camera = cv2.VideoCapture(0)
    weave.init('altryne-halloween-2024')
    if not camera.isOpened():
        raise RuntimeError("Could not open camera")

    # Initialize Porcupine
    access_key = os.getenv("PICOVOICE_ACCESS_KEY")
    if not access_key:
        raise RuntimeError("PICOVOICE_ACCESS_KEY not found in environment variables.")

    import platform
    if platform.system() == "Darwin":  # Mac OS
        keyword_path = "./porcupine_models/trick-or-treat_en_mac_v3_0_0.ppn"
    elif platform.system() == "Linux" and platform.machine().startswith("arm"):  # Raspberry Pi
        keyword_path = "./porcupine_models/trick-or-treat_en_raspberry-pi_v2_2_0.ppn"
    else:
        raise RuntimeError("Unsupported platform for wake word detection")
    porcupine = pvporcupine.create(access_key=access_key, keyword_paths=[keyword_path])

    # Initialize PvRecorder
    recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
    recorder.start()

    # Start frame capture in a separate thread
    capture_thread = Thread(target=capture_frames, daemon=True)
    capture_thread.start()

    # Start motion detection in a separate thread only if enabled
    if ENABLE_MOTION_DETECTION:
        motion_thread = Thread(target=motion_detector, daemon=True)
        motion_thread.start()

    # Start wake word detection in a separate thread
    wake_word_thread = Thread(target=wake_word_detector, daemon=True)
    wake_word_thread.start()

    try:
        yield
    finally:
        if camera:
            camera.release()
        if porcupine:
            porcupine.delete()
        if recorder:
            recorder.stop()
            recorder.delete()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "Service is running"}

def capture_frames():
    global camera, frame_queue
    while True:
        with camera_lock:
            success, frame = camera.read()
        if success:
            if frame_queue.full():
                frame_queue.get()  # Remove oldest frame if queue is full
            frame_queue.put(frame)
        time.sleep(0.01)  # Small delay to reduce CPU usage

def gen_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.01)

@app.get("/camera")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

def motion_detector():
    global motion_detected, last_motion_time, audio_playing
    prev_frame = None
    motion_threshold = 5000  # Adjust this value to change motion sensitivity
    cooldown = 5  # Cooldown period in seconds

    while True:
        if not frame_queue.empty() and not audio_playing:
            frame = frame_queue.get()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_frame is None:
                prev_frame = gray
                continue

            frame_diff = cv2.absdiff(prev_frame, gray)
            thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > motion_threshold:
                    current_time = time.time()
                    if current_time - last_motion_time > cooldown:
                        motion_detected = True
                        last_motion_time = current_time
                        # Convert the frame to PIL Image
                        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        asyncio.run(process_motion(pil_image))
                    break

            prev_frame = gray
        else:
            time.sleep(0.01)

@app.get("/motion")
def check_motion():
    global motion_detected
    if ENABLE_MOTION_DETECTION and motion_detected:
        motion_detected = False
        return {"motion": True}
    return {"motion": False}

@weave.op
async def process_motion(pil_image):
    await stream_text_to_speech("Oooh, who do we have here?")
    if CHAT_MODEL == "gemini":
        response = gemini_chat(pil_image)
    elif CHAT_MODEL == "openai":
        response = openai_chat(pil_image)
    elif CHAT_MODEL == "openrouter":
        response = openrouter_chat(pil_image)
    else:
        raise ValueError(f"Unknown chat model: {CHAT_MODEL}")

    await stream_text_to_speech(response)
    return response

async def stream_text_to_speech(text):
    global audio_playing
    client = AsyncCartesia(api_key=os.environ.get("CARTESIA_API_KEY"))
    ws = await client.tts.websocket()
    ctx = ws.context()

    # Initialize audio playback
    p = pyaudio.PyAudio()
    stream_audio = None
    rate = 44100

    voice_id = "87748186-23bb-4158-a1eb-332911b0b708"  # Replace with your actual voice ID from Cartesia
    model_id = "sonic-english"
    output_format = {
        "container": "raw",
        "encoding": "pcm_s16le",  # Changed to 16-bit PCM for better compatibility
        "sample_rate": rate,
    }

    try:
        audio_playing = True  # Set flag to True before starting audio playback
        await ctx.send(
            model_id=model_id,
            transcript=text,
            voice_id=voice_id,
            continue_=False,
            output_format=output_format,
        )

        # Iterate over the async generator to receive audio data
        async for audio_buffer in ctx.receive():
            if audio_buffer:
                # Debugging: Print the type and keys of audio_buffer
                # Remove or comment out this line in production
                # print(f"Received audio_buffer: {audio_buffer}")

                # Determine the structure of audio_buffer and extract audio bytes accordingly
                audio_data_bytes = None

                if isinstance(audio_buffer, dict):
                    # Check if 'data' key exists
                    audio_data_bytes = audio_buffer.get('data')
                    if not audio_data_bytes:
                        # If 'data' key does not exist, check other possible keys or handle accordingly
                        # For example, check if the first element is bytes
                        if len(audio_buffer) > 0 and isinstance(list(audio_buffer.values())[0], bytes):
                            audio_data_bytes = list(audio_buffer.values())[0]
                elif isinstance(audio_buffer, (tuple, list)):
                    # If audio_buffer is a tuple or list, assume the first element is bytes
                    if len(audio_buffer) > 0 and isinstance(audio_buffer[0], bytes):
                        audio_data_bytes = audio_buffer[0]
                elif isinstance(audio_buffer, bytes):
                    # If audio_buffer is bytes, use it directly
                    audio_data_bytes = audio_buffer
                else:
                    print("Unknown audio_buffer format.")
                    continue

                if not audio_data_bytes:
                    # If unable to extract bytes, skip processing
                    print("No audio bytes found in audio_buffer. Skipping...")
                    continue

                if not stream_audio:
                    stream_audio = p.open(
                        format=pyaudio.paInt16,  # Changed to match the new encoding
                        channels=1,
                        rate=rate,
                        output=True,
                        frames_per_buffer=1024,  # Adjust buffer size for smoother playback
                    )
                try:
                    # Increase volume by multiplying the audio data
                    volume_multiplier = 2.0  # Adjust this value to increase or decrease volume
                    audio_data = np.frombuffer(audio_data_bytes, dtype=np.int16)
                    audio_data = (audio_data * volume_multiplier).astype(np.int16)
                    stream_audio.write(audio_data.tobytes())
                except Exception as e:
                    print(f"Error processing audio data: {e}")
    except Exception as e:
        print(f"Error during text-to-speech: {e}")
    finally:
        if stream_audio:
            stream_audio.stop_stream()
            stream_audio.close()
        p.terminate()
        await ws.close()
        await client.close()
        audio_playing = False  # Set flag back to False after audio playback is complete

def wake_word_detector():
    global porcupine, recorder
    try:
        while True:
            pcm = recorder.read()
            result = porcupine.process(pcm)
            if result >= 0:
                print("Wake word detected!")
                # Add your wake word detection logic here
                asyncio.run(handle_wake_word())
    except KeyboardInterrupt:
        print("Wake word detection stopped.")
    except Exception as e:
        print(f"Error in wake word detector: {e}")

async def handle_wake_word():
    # take a picture
    # run the chat model
    # stream the response to the user
    
    await stream_text_to_speech("Hello! How can I assist you today?")
    # Add additional wake word handling logic here

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)