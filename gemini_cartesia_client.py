import os
import asyncio
import random
from dotenv import load_dotenv
from skeleton_control import SkeletonControl
from cartesia_client import CartesiaStreamingClient
from gemini import model, generation_config
import weave

load_dotenv()

class GeminiCartesiaClient:
    def __init__(self, skeleton: SkeletonControl = None):
        # Cartesia setup
        self.cartesia_client = CartesiaStreamingClient(skeleton=skeleton)
        self.model = model  # Use the configured model from gemini.py

    @weave.op
    async def generate_and_speak(self, num_winners=3):
        try:
            names = await asyncio.to_thread(self.load_names)  # Run in thread pool
            prompt_data = await asyncio.to_thread(self.create_prompt, names, num_winners)
            gemini_response = await self.get_gemini_response(prompt_data["prompt"])  # Use the prompt string
            await self.cartesia_client.stream_tts(gemini_response)
            return {
                "num_winners": num_winners,
                "num_names": len(names),
                "response": gemini_response
            }
        except Exception as e:
            print(f"🎃 Spooky error occurred: {e}")
            # Maybe have a fallback response ready?
            raise

    @weave.op
    def load_names(self):
        with open('names.txt', 'r') as file:
            return [line.strip() for line in file if line.strip()]

    @weave.op
    def create_prompt(self, names, num_winners):
        num_names = len(names)
        winners = random.sample(names, num_winners)
        first_letters = [name[0].upper() for name in winners]
        
        prompt = f"""You're a spooky but friendly skeleton announcing raffle winners at Seattle Tinkerers.
        
        1. Start with a spooky but friendly greeting.
        2. Mention how amazing Seattle Tinkerers is as a community of {num_names} makers and tech enthusiasts.
        3. Announce that you're selecting {num_winners} lucky winners for the Meta Raybans raffle.
        4. Say that the winners' names start with the letters {', '.join(first_letters)}.
        5. Tell everyone to check Weave to see if they won.
        
        Keep it fun and exciting, with a Halloween twist!
        """
        return {
            "prompt": prompt,
            "num_names": num_names,
            "winner_letters": first_letters
        }

    @weave.op
    async def get_gemini_response(self, prompt):
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text

    async def close(self):
        await self.cartesia_client.close()

@weave.op
async def test_gemini_cartesia():
    skeleton = SkeletonControl()
    client = GeminiCartesiaClient(skeleton=skeleton)

    try:
        skeleton.start_body_movement()
        skeleton.eyes_on()
        result = await client.generate_and_speak()
        return result
    finally:
        await client.close()
        skeleton.stop_body_movement()
        skeleton.eyes_off()

if __name__ == "__main__":
    skeleton = SkeletonControl()
    client = GeminiCartesiaClient(skeleton=skeleton)

    try:
        skeleton.start_body_movement()
        skeleton.eyes_on()
        # Just run it synchronously since it's a one-off script
        asyncio.run(client.generate_and_speak())
    finally:
        # Cleanup
        asyncio.run(client.close())
        skeleton.stop_body_movement()
        skeleton.eyes_off()
