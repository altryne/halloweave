import os
import asyncio
import numpy as np
import pyaudio
from elevenlabs.client import AsyncElevenLabs, VoiceSettings
from typing import AsyncGenerator, Dict, Union, Optional
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import pygame
from io import BytesIO
import wave
from pydub import AudioSegment

from skeleton_control import SkeletonControl
import time

load_dotenv()


class ElevenLabsStreamingClient:
    def __init__(self, skeleton: Optional[SkeletonControl] = None):
        self.api_key = os.environ.get("ELEVEN_API_KEY")
        self.client = AsyncElevenLabs(api_key=self.api_key)
        self.voice_id = "kiJFNumu1N2HlJRDOwrv"
        self.model_id = "eleven_turbo_v2_5"
        self.rate = 44100
        self.skeleton = skeleton
        self.p = pyaudio.PyAudio()
        self.audio_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            output=True,
            frames_per_buffer=4096
        )
        self.executor = ThreadPoolExecutor(max_workers=2)
        pygame.mixer.init(frequency=44100)  # Match sample rate
        self.voice_settings = VoiceSettings(
            stability=0.8,  # 80%
            similarity_boost=0.63,  # 63%
            style=0.47,  # 47%
            use_speaker_boost=True
        )

    async def stream_tts(self, text: str):
        try:
            # Request MP3 format
            response = self.client.text_to_speech.convert(
                voice_id=self.voice_id,
                output_format="mp3_44100_128",
                text=text,
                model_id=self.model_id,
                voice_settings=self.voice_settings,  # Use the settings from init
            )

            # Create a BytesIO buffer for the MP3 data
            audio_buffer = BytesIO()
            
            # Collect all MP3 data first
            async for chunk in response:
                if chunk:
                    audio_buffer.write(chunk)
            
            audio_buffer.seek(0)
            
            # Convert MP3 to AudioSegment
            audio = AudioSegment.from_mp3(audio_buffer)
            
            # Export as WAV to a new BytesIO buffer (this ensures proper PCM format)
            wav_buffer = BytesIO()
            audio.export(wav_buffer, format='wav')
            wav_buffer.seek(0)
            
            # Read WAV file
            with wave.open(wav_buffer, 'rb') as wf:
                # Read data
                audio_data = wf.readframes(wf.getnframes())
                # Convert to numpy array
                samples = np.frombuffer(audio_data, dtype=np.int16)
                
                # Process in larger chunks
                chunk_size = 4096
                for i in range(0, len(samples), chunk_size):
                    chunk = samples[i:i + chunk_size]
                    if len(chunk) > 0:
                        # Play audio
                        self.audio_stream.write(chunk.tobytes())
                        
                        # Move skeleton mouth
                        if self.skeleton:
                            loop = asyncio.get_running_loop()
                            await loop.run_in_executor(
                                self.executor,
                                self.skeleton.move_mouth,
                                chunk.tobytes()
                            )

        except Exception as e:
            print(f"Error in stream_tts: {e}")
            raise e
        finally:
            if self.skeleton:
                self.skeleton.stop_mouth_movement()

    async def _handle_chunk(self, chunk: Dict[str, Union[bytes, float]]):
        try:
            # Convert bytes to numpy array of int16
            audio_data = np.frombuffer(chunk['audio'], dtype=np.int32)
            
            # Normalize the int32 data to float32 in the range [-1, 1]
            audio_float = audio_data.astype(np.float32) / np.iinfo(np.int32).max
            
            # Convert float32 to int16
            audio_int16 = (audio_float * 32767).astype(np.int16)
            
            # Apply volume multiplier
            volume_multiplier = 1
            audio_int16 = (audio_int16 * volume_multiplier).astype(np.int16)
            
            # Convert to bytes for playback
            audio_bytes = audio_int16.tobytes()

            # Play audio
            self.audio_stream.write(audio_bytes)

            # Move skeleton mouth if instance is provided
            if self.skeleton:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self.executor, self.skeleton.move_mouth, audio_bytes)

            print(f"Timestamp: {chunk['timestamp']:.2f}s")
        except Exception as e:
            print(f"Error handling chunk: {e}")

    async def close(self):
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.skeleton:
            self.skeleton.stop_mouth_movement()
        self.p.terminate()
        self.executor.shutdown()

async def test_streaming():
    skeleton = SkeletonControl()
    client = ElevenLabsStreamingClient(skeleton=skeleton)

    try:
        skeleton.start_body_movement()
        skeleton.eyes_on()

        text = "Welcome, welcome mortals to my humble abode. Now who do we have here? A witch? That's so awesome."
        await client.stream_tts(text)
    finally:
        await client.close()
        skeleton.stop_body_movement()
        skeleton.eyes_off()

if __name__ == "__main__":
    asyncio.run(test_streaming())
