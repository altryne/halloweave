import asyncio
import os
from openai_realtime_client import RealtimeClient, AudioHandler
from llama_index.core.tools import FunctionTool
import dotenv
from skeleton_control import SkeletonControl

dotenv.load_dotenv()

# Try to import InputHandler, create a dummy class if it fails
try:
    from openai_realtime_client.handlers import InputHandler
except ImportError:
    class InputHandler:
        def __init__(self):
            self.command_queue = asyncio.Queue()

        async def on_press(self, key):
            pass  # Dummy method

def get_phone_number(name: str) -> str:
    """Get my phone number."""
    if name == "Jerry":
        return "1234567890"
    elif name == "Logan":
        return "0987654321"
    else:
        return "Unknown"

tools = [FunctionTool.from_defaults(fn=get_phone_number)]

class RealtimeAPI:
    def __init__(self):
        self.client = None
        self.audio_handler = AudioHandler()
        self.connection_active = False
        self.skeleton = SkeletonControl()

    async def connect(self):
        if self.connection_active:
            return

        self.client = RealtimeClient(
            api_key=os.environ.get("OPENAI_API_KEY"),
            on_text_delta=lambda text: print(f"\nAssistant: {text}", end="", flush=True),
            on_audio_delta=self.handle_audio_delta,
            tools=tools,
        )
        await self.client.connect()
        self.connection_active = True
        print("Connected to OpenAI Realtime API!")

    async def disconnect(self):
        if not self.connection_active:
            return

        await self.client.close()
        self.connection_active = False
        print("Disconnected from OpenAI Realtime API!")

    def handle_audio_delta(self, audio):
        # Play the audio
        self.audio_handler.play_audio(audio)
        
        # Move the skeleton's mouth
        self.skeleton.move_mouth(audio)

    async def record_and_send_audio(self, duration=5):
        print(f"Recording audio for {duration} seconds...")
        self.audio_handler.start_recording()
        await asyncio.sleep(duration)
        audio_data = self.audio_handler.stop_recording()
        print("Recording finished.")
        
        if audio_data and self.connection_active:
            print(f"Audio data length: {len(audio_data)} bytes")
            await self.client.send_audio(audio_data)
            print("Audio sent to Realtime API")
        else:
            print("No audio data recorded or connection is not active")

    async def run(self):
        try:
            await self.connect()
            
            # Start message handling in the background
            message_handler = asyncio.create_task(self.client.handle_messages())
            
            # Record and send audio
            await self.record_and_send_audio()
            
            # Keep the connection alive
            while True:
                await asyncio.sleep(1)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            # Clean up
            self.audio_handler.cleanup()
            self.skeleton.cleanup()
            await self.disconnect()

realtime_api = RealtimeAPI()

if __name__ == "__main__":
    print("Starting Realtime API...")
    asyncio.run(realtime_api.run())