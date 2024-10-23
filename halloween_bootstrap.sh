#!/bin/bash

# Export XDG_RUNTIME_DIR for PulseAudio
export XDG_RUNTIME_DIR=/run/user/$(id -u altryne)

# Wait for audio device to be ready
sleep 5

# Ensure PulseAudio is running for this user
pulseaudio --start

# Connect to the Bluetooth device
for attempt in {1..3}; do
    if bluetoothctl connect D6:F7:3D:E8:F9:68; then
        echo "Bluetooth device connected successfully."
        break
    else
        echo "Failed to connect to Bluetooth device. Attempt $attempt of 3."
        if [ $attempt -lt 3 ]; then
            echo "Waiting 3 seconds before retrying..."
            sleep 3
        else
            echo "Failed to connect after 3 attempts. Continuing with the script."
        fi
    fi
done

# Wait for the Bluetooth connection to establish
sleep 2

# Change to the Halloween project directory
cd /home/altryne/halloweave

# Activate the virtual environment
source .venv/bin/activate

# Run diagnostics (skeleton_control.py)
python skeleton_control.py

# Run the main application
python main.py
