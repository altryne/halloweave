import os
import asyncio
import numpy as np
import pyaudio
from cartesia import AsyncCartesia
from typing import AsyncGenerator, Dict, Union, Optional
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from skeleton_control import SkeletonControl
import time 
import pygame
load_dotenv()

class CartesiaStreamingClient:
    def __init__(self, skeleton: Optional[SkeletonControl] = None):
        self.api_key = os.environ.get("CARTESIA_API_KEY")
        self.client = AsyncCartesia(api_key=self.api_key)
        self.voice_id = "87748186-23bb-4158-a1eb-332911b0b708"
        self.model_id = "sonic-english"
        self.rate = 22050
        self.skeleton = skeleton
        self.audio_playing = False
        self.p = pyaudio.PyAudio()
        self.audio_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            output=True,
            frames_per_buffer=4096,
        )
        self.executor = ThreadPoolExecutor(max_workers=2)
        pygame.mixer.init()

    async def stream_tts(self, text: str, use_sse: bool = False):
        output_format = {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": self.rate,
        }

        if use_sse:
            async for chunk in self._stream_sse(text, output_format):
                await self._handle_chunk(chunk)
        else:
            async for chunk in self._stream_websocket(text, output_format):
                await self._handle_chunk(chunk)

    async def _stream_sse(self, text: str, output_format: Dict) -> AsyncGenerator[Dict[str, Union[bytes, float]], None]:
        async with self.client.tts.stream(
            model_id=self.model_id,
            transcript=text,
            voice_id=self.voice_id,
            output_format=output_format,
        ) as response:
            start_time = asyncio.get_event_loop().time()
            async for chunk in response:
                timestamp = asyncio.get_event_loop().time() - start_time
                yield {"audio": chunk, "timestamp": timestamp}

    async def _stream_websocket(self, text: str, output_format: Dict) -> AsyncGenerator[Dict[str, Union[bytes, float]], None]:
        ws = await self.client.tts.websocket()
        ctx = ws.context()

        try:
            await ctx.send(
                model_id=self.model_id,
                transcript=text,
                voice_id=self.voice_id,
                continue_=False,
                add_timestamps=True,
                output_format=output_format,
            )
            

            start_time = asyncio.get_event_loop().time()
            # Import pygame and initialize the mixer
            


            def fade_music(target_volume, duration_ms, steps=100):
                """Fade the music volume to the target_volume over duration_ms milliseconds."""
                current_volume = pygame.mixer.music.get_volume()
                volume_difference = current_volume - target_volume
                delay = duration_ms / steps / 1000.0  # Convert milliseconds to seconds
                for i in range(steps):
                    # Calculate the new volume
                    new_volume = current_volume - (volume_difference * (i + 1) / steps)
                    pygame.mixer.music.set_volume(new_volume)
                    time.sleep(delay)

            fade_music(0.3, 1000)

            async for response in ctx.receive():
                
                if response.get('audio'):
                    audio_data_bytes = self._extract_audio_bytes(response['audio'])
                    if audio_data_bytes:
                        timestamp = asyncio.get_event_loop().time() - start_time
                        yield {"audio": audio_data_bytes, "timestamp": timestamp}
        finally:
            await ws.close()
            await self.client.close()

    @staticmethod
    def _extract_audio_bytes(audio_buffer):
        if isinstance(audio_buffer, dict):
            return audio_buffer.get('data') or next((v for v in audio_buffer.values() if isinstance(v, bytes)), None)
        elif isinstance(audio_buffer, (tuple, list)) and audio_buffer and isinstance(audio_buffer[0], bytes):
            return audio_buffer[0]
        elif isinstance(audio_buffer, bytes):
            return audio_buffer
        return None

    async def _handle_chunk(self, chunk: Dict[str, Union[bytes, float]]):
        audio_data = np.frombuffer(chunk['audio'], dtype=np.int16)
        volume_multiplier = 1.5
        audio_data = (audio_data * volume_multiplier).astype(np.int16)

        # Play audio
        self.audio_stream.write(audio_data.tobytes())

        # Move skeleton mouth if instance is provided
        if self.skeleton:
            if asyncio.iscoroutinefunction(self.skeleton.move_mouth):
                await self.skeleton.move_mouth(audio_data.tobytes())
            else:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self.executor, self.skeleton.move_mouth, audio_data.tobytes())

        # print(f"Timestamp: {chunk['timestamp']:.2f}s")

    async def close(self):
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        self.p.terminate()
        self.executor.shutdown()
        if self.client:
            await self.client.close()

async def test_streaming(use_sse: bool = False):
    skeleton = SkeletonControl()
    client = CartesiaStreamingClient(skeleton=skeleton)

    try:
        skeleton.start_body_movement()
        skeleton.eyes_on()

        text = "Hello, this is a tes. How does it sound?"
        await client.stream_tts(text, use_sse)
    finally:
        await client.close()
        skeleton.stop_body_movement()
        skeleton.eyes_off()

if __name__ == "__main__":
    import sys
    use_sse = "--sse" in sys.argv
    asyncio.run(test_streaming(use_sse))