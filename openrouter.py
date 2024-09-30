import os
import requests
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def openrouter_chat(image_path):
    base64_image = encode_image(image_path)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"
    }

    payload = {
        "model": "anthropic/claude-3-opus-20240229",  # You can change this to any model supported by OpenRouter
        "messages": [
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
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code} - {response.text}"