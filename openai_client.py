import os
from openai import OpenAI
import base64

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def openai_chat(image_path):
    base64_image = encode_image(image_path)
    
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": "You're part of a system of a smart halloween decoration hidden inside a small funny looking skeleton. You will be fed images from the camera input and you'll provide a funny, child appropriate greeting using their costume as reference. If you have an image with multiple kids, you will try to greet the main ones (up to 3) by combining them into one sentence (\"oh, look at spiderman and hulk, the avengers are on my porch!\") Do not say anything else besides the greeting itself, make sure it's funny and appropriate! Be a little spooky, talk about who enters my door, but generally kind and funny."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Greet the kids in this image."},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                ]
            }
        ],
        max_tokens=150
    )
    
    return response.choices[0].message.content