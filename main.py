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

import pyaudio
from PIL import Image
import numpy as np
import cv2

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

# Add this import at the top of the file, with the other imports
from camera_module import Camera, get_camera

# Add these imports at the top of the file
# from livekit_agent import initialize_livekit_agent, HalloweenAgent
# from livekit import rtc
import json
import openai
from skeleton_control import SkeletonControl

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

# Add these global variables
CONVERSATION_MODE = "regular"  # Options: "regular", "live"
livekit_room = None
livekit_agent = None

# Add this global variable
skeleton = None

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global camera, porcupine, recorder, skeleton
    # Replace the existing camera initialization with:
    camera = get_camera()
    
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
    else:
        keyword_path = "./porcupine_models/trick-or-treat_en_raspberry-pi_v2_2_0.ppn"
    
    porcupine = pvporcupine.create(access_key=access_key, keyword_paths=[keyword_path])

    # Initialize PvRecorder
    recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
    recorder.start()

    # Initialize skeleton control
    skeleton = SkeletonControl()

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
        if CONVERSATION_MODE == "live" and livekit_room:
            await livekit_room.disconnect()
        if skeleton:
            skeleton.cleanup()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "Service is running"}

def capture_frames():
    global camera, frame_queue
    while True:
        with camera_lock:
            frame = camera.capture_image()  # Use capture_frame() instead of read()
        if frame is not None:
            if frame_queue.full():
                frame_queue.get()  # Remove oldest frame if queue is full
            frame_queue.put(frame)
        time.sleep(0.01)  # Small delay to reduce CPU usage

def gen_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            # Assuming frame is already in the correct format (JPEG bytes)
            # If not, you may need to convert it here
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
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
            # Assuming frame is a PIL Image, convert it to numpy array for OpenCV operations
            frame_np = np.array(frame)
            gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
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
                        # frame is already a PIL Image, so no need to convert
                        asyncio.run(process_motion(frame))
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
    global audio_playing, skeleton
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
        skeleton.start_mouth_movement()  # Start moving the skeleton's mouth
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
                    volume_multiplier = 1.5  # Adjust this value to increase or decrease volume
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
        skeleton.stop_mouth_movement()  # Stop moving the skeleton's mouth

def wake_word_detector():
    global porcupine, recorder, skeleton
    try:
        while True:
            pcm = recorder.read()
            result = porcupine.process(pcm)
            if result >= 0:
                print("Wake word detected!")
                skeleton.eyes_on()  # Turn on skeleton eyes
                if CONVERSATION_MODE == "live":
                    asyncio.run(handle_wake_word_live())
                else:
                    asyncio.run(handle_wake_word())
                skeleton.eyes_off()  # Turn off skeleton eyes after conversation
    except KeyboardInterrupt:
        print("Wake word detection stopped.")
    except Exception as e:
        print(f"Error in wake word detector: {e}")

# Add a new function to handle wake word in live mode
async def handle_wake_word_live():
    global camera, livekit_room, livekit_agent
    print("Wake word detected! Starting live conversation and taking a picture...")
    
    # Initialize LiveKit room and agent when wake word is detected
    if CONVERSATION_MODE == "live" and not livekit_room:
        try:
            livekit_room, livekit_agent = await initialize_livekit_agent()
            print("LiveKit room and agent initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize LiveKit room and agent: {e}")
            return

    # Capture an image
    pil_image = camera.capture_image()
    
    if pil_image is not None:
        # Process the image with Gemini in the background
        gemini_thread = Thread(target=process_image_with_gemini, args=(pil_image,), daemon=True)
        gemini_thread.start()
        
        # Start the LiveKit conversation
        if livekit_agent:
            await livekit_agent.on_start()
        else:
            print("LiveKit agent not initialized. Cannot start conversation.")
        
        # Start listening for audio input
        audio_input_thread = Thread(target=listen_for_audio_input, daemon=True)
        audio_input_thread.start()
    else:
        print("Failed to capture an image.")

# Add a new function to listen for audio input
def listen_for_audio_input():
    global recorder, livekit_agent
    try:
        while True:
            audio_data = recorder.read()
            asyncio.run(livekit_agent.on_audio(audio_data))
    except Exception as e:
        print(f"Error listening for audio input: {e}")

# Add a new function to process the image with Gemini
def process_image_with_gemini(pil_image):
    prompt = "Identify the costume/costumers in this image and respond with a detailed description of the image with emphasis on Halloween and costumes. If there's no costume detected, reply with 'no costume'. If there are multiple ones, or you think this is a family, say a dad with a kid, respond with that info."
    response = gemini_chat(pil_image, prompt)
    print(f"Gemini image analysis: {response}")
    # You can add logic here to use the Gemini response if needed

# Update the handle_wake_word function for regular mode
async def handle_wake_word():
    global camera
    #play sound from sounds directory using pyaudio
    def play_sound():
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, output=True, frames_per_buffer=1024)
        with open("sounds/boogeyman.wav", "rb") as f:
            data = f.read()
        stream.write(data)
        stream.stop_stream()
        stream.close()
        p.terminate()

    sound_thread = threading.Thread(target=play_sound)
    sound_thread.start()
    print("Wake word detected! Taking a picture...")
    
    pil_image = camera.capture_image()
    
    if pil_image is not None:
        if CHAT_MODEL == "gemini":
            response = gemini_chat(pil_image)
        elif CHAT_MODEL == "openai":
            response = openai_chat(pil_image)
        elif CHAT_MODEL == "openrouter":
            response = openrouter_chat(pil_image)
        else:
            response = "I'm sorry, but I couldn't process the image at the moment."
        
        
        await stream_text_to_speech(response)
    else:
        await stream_text_to_speech("I'm sorry, but I couldn't take a picture at the moment.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)