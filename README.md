# Halloweave 🎃

An AI-powered Halloween decoration that detects motion, analyzes costumes, and provides spooky greetings!

## 🔗 Project Links

- GitHub Repository: [https://github.com/altryne/halloweave](https://github.com/altryne/halloweave)
- Project Announcement: [Twitter Post](https://x.com/altryne/status/1840724089981251989)

## 🚀 Features

- Motion detection using OpenCV
- Costume analysis with Gemini AI
- Spooky text-to-speech using Cartesia API
- FastAPI web server for easy integration and monitoring
- Experiment tracking with Weights & Biases (Weave)

## 📋 Prerequisites

- Python 3.8+
- OpenCV
- PyAudio
- FastAPI
- Uvicorn
- Pillow (PIL)
- Weave
- Cartesia
- Google Generative AI

## 🛠 Installation

1. Clone the repository:
   ```
   git clone https://github.com/altryne/halloweave.git
   cd halloweave
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables in a `.env` file:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   CARTESIA_API_KEY=your_cartesia_api_key
   WANDB_API_KEY=your_wandb_api_key
   ```

## 🎯 Usage

Run the main application:
```
python main.py
```

## 📷 Camera Setup

Ensure your camera is correctly configured and accessible by OpenCV. You may need to adjust the camera index (`0` by default) in the code if your system has multiple cameras.

