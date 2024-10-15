from fastapi import APIRouter, HTTPException, Request, Depends, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
# from realtime_api import realtime_api
from skeleton_control import SkeletonControl
from image_handler import sse
import json
from sse_manager import sse_endpoint

router = APIRouter()

def get_skeleton(request: Request):
    return request.app.state.skeleton

class WebhookPayload(BaseModel):
    button: int

@router.post("/webhook")
async def webhook(payload: WebhookPayload):
    print(payload)
    valid_values = [0, 1, 2, 4, 8, 16]
    if payload.button not in valid_values:
        raise HTTPException(status_code=400, detail="Invalid button value")
    holding_16 = False
    holding_8 = False
    holding_4 = False
    if payload.button == 1:
        # Trigger the same action as when a wake word is detected
        from main import handle_wake_word
        asyncio.create_task(handle_wake_word())
        return {"message": "Wake word action triggered"}
    elif payload.button == 0:
        return {"message": "0 called"}
    elif payload.button == 8:
        return {"message": "Action for button 8"}
    elif payload.button == 16:
        return {"message": "Action for button 16"}
    elif payload.button == 2:
        # if realtime_api.connection_active:
        #     await realtime_api.disconnect()
        #     return {"message": "Realtime API connection closed"}
        # else:
        #     await realtime_api.connect()
        #     await realtime_api.run()
        #     return {"message": "Realtime API connection opened"}
        print("Realtime API connection opened")
    elif payload.button == 4:
        # if not holding_4:
        #     holding_4 = True
        #     await realtime_api.record_and_send_audio()
        #     return {"message": "Started and finished listening to microphone"}
        # else:
        #     holding_4 = False
        #     return {"message": "Stopped listening to microphone"}
        print("stopped listening to microphone")

class CommandPayload(BaseModel):
    command: str

@router.post("/commands")
async def handle_command(request: Request, skeleton: SkeletonControl = Depends(get_skeleton)):
    content_type = request.headers.get("Content-Type", "")
    
    if "application/json" in content_type:
        payload = await request.json()
        command = payload.get("command")
    else:
        form_data = await request.form()
        command = form_data.get("command")
    
    if not command:
        raise HTTPException(status_code=400, detail="Missing command")

    if command == "toggle_eyes":
        skeleton.toggle_eyes()
        return {"message": "Eyes toggled"}
    elif command == "toggle_mouth":
        skeleton.toggle_mouth()
        return {"message": "Mouth toggled"}
    elif command == "toggle_both":
        skeleton.toggle_both()
        return {"message": "Both toggled"}
    elif command == "trigger_wake_word":
        from main import handle_wake_word
        asyncio.create_task(handle_wake_word())
        return {"message": "Wake word triggered manually"}
    else:
        raise HTTPException(status_code=400, detail="Invalid command")

async def status_event_generator(request: Request):
    skeleton = request.app.state.skeleton
    while True:
        data = {
            "type": "status",
            "data": {
                "eyes": "Active" if skeleton.eyes_lit else "Inactive",
                "mouth": "Active" if skeleton.mouth_speaking else "Inactive",
                "body": "Active" if skeleton.body_moving else "Inactive",
                "both": "Active" if skeleton.eyes_lit and skeleton.mouth_speaking else "Inactive"
            }
        }
        yield json.dumps(data)
        await asyncio.sleep(1)  # Send updates every second

@router.get("/sse")
async def sse(request: Request):
    return await sse_endpoint(request)
