from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import asyncio

router = APIRouter()

class WebhookPayload(BaseModel):
    button: int

@router.post("/webhook")
async def webhook(payload: WebhookPayload):
    
    valid_values = [0, 1, 2, 4, 8, 16]
    if payload.button not in valid_values:
        raise HTTPException(status_code=400, detail="Invalid button value")

    if payload.button == 1:
        # Trigger the same action as when a wake word is detected
        from main import handle_wake_word
        asyncio.create_task(handle_wake_word())
        return {"message": "Wake word action triggered"}
    elif payload.button == 0:
        return 
    elif payload.button == 2:
        return {"message": "Action for button 2"}
    elif payload.button == 4:
        return {"message": "Action for button 4"}
    elif payload.button == 8:
        return {"message": "Action for button 8"}
    elif payload.button == 16:
        return {"message": "Action for button 16"}