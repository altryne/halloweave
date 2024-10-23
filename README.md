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

## 🤖 Automatic Service Setup (Raspberry Pi)

To run the Halloween project automatically on your Raspberry Pi, follow these steps:

1. Create a symbolic link for the service file in the systemd directory:
   ```
   sudo ln -s /home/altryne/halloweave/halloween.service /etc/systemd/system/halloween.service
   ```

2. Make sure the bootstrap script is executable:
   ```
   chmod +x /home/altryne/halloweave/halloween_bootstrap.sh
   ```

3. Reload the systemd daemon:
   ```
   sudo systemctl daemon-reload
   ```

4. Enable the service to start on boot:
   ```
   sudo systemctl enable halloween.service
   ```

5. Start the service:
   ```
   sudo systemctl start halloween.service
   ```

### Managing the Service

- To stop the service:
  ```
  sudo systemctl stop halloween.service
  ```

- To restart the service:
  ```
  sudo systemctl restart halloween.service
  ```

- To check the status of the service:
  ```
  sudo systemctl status halloween.service
  ```

- To view the service logs:
  ```
  sudo journalctl -u halloween.service
  ```

## 📷 Camera Setup

Ensure your camera is correctly configured and accessible by OpenCV. You may need to adjust the camera index (`0` by default) in the code if your system has multiple cameras.

