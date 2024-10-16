from fastapi import Request
from sse_starlette.sse import EventSourceResponse
import asyncio

class SSEManager:
    def __init__(self):
        self.clients = set()

    async def broadcast(self, message: str):
        for client in self.clients:
            await client.send(message)

    async def listen(self, request: Request):
        client = SSEClient(request)
        self.clients.add(client)
        try:
            yield client.listen()
        finally:
            self.clients.remove(client)

class SSEClient:
    def __init__(self, request: Request):
        self.request = request
        self.queue = asyncio.Queue()

    async def send(self, message: str):
        await self.queue.put(message)

    async def listen(self):
        while True:
            if await self.request.is_disconnected():
                break
            message = await self.queue.get()
            yield message

async def sse_endpoint(request: Request):
    manager = request.app.state.sse_manager
    return EventSourceResponse(manager.listen(request))
