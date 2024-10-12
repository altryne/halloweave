from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import asyncio
from realtime_api import realtime_api

router = APIRouter()

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
        if realtime_api.connection_active:
            await realtime_api.disconnect()
            return {"message": "Realtime API connection closed"}
        else:
            await realtime_api.connect()
            await realtime_api.run()
            return {"message": "Realtime API connection opened"}
    elif payload.button == 4:
        if not holding_4:
            holding_4 = True
            # Change this line to await the result
            await realtime_api.record_and_send_audio()
            return {"message": "Started and finished listening to microphone"}
        else:
            holding_4 = False
            return {"message": "Stopped listening to microphone"}