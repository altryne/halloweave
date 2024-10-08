#!/bin/bash



# Connect to the Bluetooth device
for attempt in {1..3}; do
    if bluetoothctl connect FC:A1:83:5E:5B:43; then
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
# sleep 2

# Change to the Halloween project directory
cd /home/altryne/halloween

# Activate the virtual environment
source .venv/bin/activate

# Run diagnostics (skeleton_control.py)
python skeleton_control.py

# Run the main application
python main.py
