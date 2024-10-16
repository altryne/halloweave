import os
from datetime import datetime
from fastapi import HTTPException
from sse_starlette.sse import EventSourceResponse
import json
import asyncio
from PIL import Image, ImageDraw

class ImageHandler:
    def __init__(self, static_folder="static"):
        self.static_folder = static_folder
        self.image_path = os.path.join(static_folder, "taken_image.jpg")
        # Create a default image if it doesn't exist
        if not os.path.exists(self.image_path):
            img = Image.new('RGB', (300, 300), color = (73, 109, 137))
            d = ImageDraw.Draw(img)
            d.text((10,10), "No image yet", fill=(255,255,0))
            img.save(self.image_path)

    def get_image_last_modified(self):
        try:
            return os.path.getmtime(self.image_path)
        except OSError:
            return 0

    def save_image(self, pil_image):
        try:
            pil_image.save(self.image_path)
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False

    async def status_event_generator(self, skeleton=None):
        last_modified = 0
        while True:
            current_modified = self.get_image_last_modified()
            data = {
                "type": "status",
                "data": {
                    "eyes": "Unknown",
                    "mouth": "Unknown",
                    "body": "Unknown"
                }
            }
            if skeleton:
                data["data"] = {
                    "eyes": "Active" if skeleton.eyes_on else "Inactive",
                    "mouth": "Active" if skeleton.mouth_speaking else "Inactive",
                    "body": "Active" if skeleton.body_moving else "Inactive"
                }
            if current_modified > last_modified:
                data["type"] = "image"
                data["data"]["imagePath"] = f"/static/taken_image.jpg?t={current_modified}"
                data["data"]["timestamp"] = datetime.fromtimestamp(current_modified).strftime("%Y-%m-%d %H:%M:%S")
                last_modified = current_modified
            
            yield json.dumps(data)
            await asyncio.sleep(1)  # Send updates every second

image_handler = ImageHandler()

async def sse(request):
    skeleton = getattr(request.app.state, 'skeleton', None)
    return EventSourceResponse(image_handler.status_event_generator(skeleton))
