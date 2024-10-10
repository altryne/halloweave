import os
import google.generativeai as genai
from PIL import Image
from google.api_core import exceptions, retry
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import weave
from dotenv import load_dotenv
import tempfile
load_dotenv()

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


def upload_to_gemini(pil_image, mime_type=None):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_filename = temp_file.name
    
    # Save the PIL image to the temporary file
    pil_image.save(temp_filename, format="PNG")
    
    try:
        # Upload the temporary file using its path
        file = genai.upload_file(temp_filename, mime_type=mime_type or "image/png")
        
        print(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file
    finally:
        # Clean up the temporary file
        os.unlink(temp_filename)


# def upload_to_cloudflare(pil_image, mime_type=None):
    
#     url = f"https://api.cloudflare.com/client/v4/accounts/{os.environ.get('CLOUDFLARE_ACCOUNT_ID')}/images/v1"
    
#     # Convert PIL Image to bytes
#     image_byte_array = io.BytesIO()
#     pil_image.save(image_byte_array, format='PNG')
#     image_byte_array = image_byte_array.getvalue()
    
#     files = {'file': ('image.png', image_byte_array, 'image/png')}
#     headers = {'Authorization': f'Bearer {os.environ.get("CLOUDFLARE_API_TOKEN")}'}
    
#     response = requests.post(url, files=files, headers=headers)
#     res = response.json()
#     if res.get("success", False):
#         return res.get("result", {}).get("variants", [None])[0]
#     else:
#         raise Exception(f"Failed to upload image to Cloudflare: {res}")

# # Usage

generation_config = {
    "temperature": 1.2,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-002",
    # model_name="gemini-1.5-flash-8b",
    generation_config=generation_config,
)

@weave.op
@retry.Retry(predicate=retry.if_exception_type(exceptions.ResourceExhausted))
def gemini_chat(pil_image):
    gemini_file = upload_to_gemini(pil_image)
    print(gemini_file)

    response = model.generate_content([
        f"""You're part of a system of a smart halloween decoration hidden inside a small funny looking skeleton. You will be fed images from the camera input and you'll provide a funny, child appropriate greeting using their costume as reference. If you have an image with multiple kids, you will try to greet the main ones (up to 3) by combining them into one sentence (\"oh, look at spiderman and hulk, the avengers are on my porch! Would you like a trick or a treat?\") \n\nDo not say anything else besides the greeting itself, make sure it's funny and appropriate! \nLook at this image and answer with a few sentences greeting "to the kid/kids. Be a little spooky, talk about who enters my door, but generally kind and funny and say at least three sentences.
        You never sayh the phrase "trick or treat", you never ask question in the end, you only answer with a greeting and an invitation to have fun on halloween. You never format or add anything to your answer, you only reply with the greeting.
        """,
        gemini_file,
    ],
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
    })
    return response.text



if __name__ == "__main__":
    img = Image.open("taken_image.jpg")
    print(upload_to_cloudflare(img))