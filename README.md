# IRP: Hand Tracking and Servo Control with Python, OpenCV, Mediapipe, and Arduino
## Overview
Python scripts developed for Individual Research Project: Year 3 Uni of Bristol
(images in folder)
This project combines computer vision-based hand tracking with Arduino-controlled servo motors to create an interactive system. Using the OpenCV and Mediapipe libraries, the system detects hand landmarks from a webcam feed, computes distances between hand features, and maps these to an array of virtual grid cells. If hand movements are detected within a certain threshold of a grid cell, the corresponding servo motor is activated via an Arduino. This repository has several iterations of the prototype's system and a heatmap script which displays the accuracy of the hand depiction from the script.

## Features:
- Hand Tracking: The program uses Mediapipe’s hand landmarks to track the movement of hands in real-time.
- Grid Mapping: A user-defined grid is displayed over the video feed, and hand positions are mapped to this grid.
- Distance Calculation: The system calculates distances from hand landmarks and hand connections to the grid’s center points to determine which grid cells are being influenced by hand movements.
- Servo Control: Based on hand movements and proximity to grid cells, servos connected to an Arduino board are activated or deactivated.
- Perpendicular Distance Calculation: A utility function computes the perpendicular distance from hand connections to grid center points to help guide the decision for servo control.

## Prerequisites:
Python Libraries:
- OpenCV (cv2)
- Mediapipe (mediapipe)
- PyFirmata (pyfirmata)
- NumPy (numpy)
- Math
  
## Hardware:
- Webcam for capturing hand movements
- Arduino board or PCA9685 board (not tested with PCA9685)
- Servo motors

## Setup Instructions
- Install Dependencies: Install the required Python libraries using pip:
- Copy code
- pip install opencv-python mediapipe pyfirmata numpy

Hardware Setup:
- Connect the Arduino to your system via a serial port (update the port variable with the correct port, e.g., 'com6').
- Connect servos to the appropriate digital pins on the Arduino.
- Run the Code:
  - Provide the grid dimensions as input in the format "width height", e.g., 8 6 for an 8x6 grid.The program will start the webcam feed and display the real-time hand-tracking results, overlaying a grid.
  - Servos will be activated or deactivated based on the proximity of hand landmarks to grid cell centers.
    
## Key Functions
- Hand Tracking: Uses Mediapipe to detect hand landmarks from the webcam feed, converting the image to RGB for landmark detection and back to BGR for display.
- Grid Mapping: Divides the video feed into a user-specified grid, draws grid lines, and calculates center points for each grid cell.
- Distance Calculations: Calculates the Euclidean distance between hand landmarks and grid center points. A threshold determines whether a hand is "close enough" to trigger servo action.
- Servo Control: The program sends signals to the Arduino board to rotate the connected servos based on the grid cell being affected by hand movements.

## How It Works
- Camera Input: The webcam captures frames at 1080p resolution.
- Hand Landmark Detection: Mediapipe detects up to 21 hand landmarks in each frame, tracking their positions.
- Grid and Distance Calculation: The grid is drawn on the video feed, and the system calculates the distances between hand landmarks and grid centers.
- Servo Activation: If a hand landmark or connection is within the threshold distance from a grid center, the corresponding servo is rotated to an angle.
  
## Notes
The grid size and servo setup are customizable depending on your project needs.
Ensure that the Arduino and servo connections are secure and that the correct port is specified for the Arduino.
The system is designed to operate with a reduced frame rate for better performance with servo control.

## Future Improvements
Optimize the frame rate and delay for smoother real-time performance.
Implement more complex gestures to control different aspects of the servos or other robotic systems.
Optimise code
Increase the number of landmarks on the hand detection to make it more accurately detect the complex shape of a hand.
