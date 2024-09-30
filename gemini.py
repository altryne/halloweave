import os
import google.generativeai as genai
import weave
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

@weave.op()
def upload_to_gemini(path, mime_type=None):
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

generation_config = {
    "temperature": 1.2,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-002",
    generation_config=generation_config,
)

@weave.op
def gemini_chat(image_path):
    file = upload_to_gemini(image_path, mime_type="image/jpeg")
    
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    file,
                    "You're part of a system of a smart halloween decoration hidden inside a small funny looking skeleton. You will be fed images from the camera input and you'll provide a funny, child appropriate greeting using their costume as reference. If you have an image with multiple kids, you will try to greet the main ones (up to 3) by combining them into one sentence (\"oh, look at spiderman and hulk, the avengers are on my porch!\") \n\nDo not say anything else besides the greeting itself, make sure it's funny and appropriate! \nLook at this image and answer with a few sentences greeting to the kid/kids. Be a little spooky, talk about who enters my door, but generally kind and funny.",
                ],
            },
        ]
    )

    response = chat_session.send_message("Greet the kids in the image.")
    return response.text