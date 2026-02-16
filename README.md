# PTZoptics_JS <img align="right" width="120" height="120" alt="ico" src="assets/app.ico" />

<img width="412.5" height="337.5" alt="image" src="https://github.com/user-attachments/assets/d9b2dff9-c12e-4029-943f-d4cb852d1583" /> <img width="412.5" height="337.5" alt="image" src="https://github.com/user-attachments/assets/d3cae73f-41c4-4991-bf50-9ae37e6d8eaf" />



**PTZoptics_JS** is a desktop application for controlling
VISCA-compatible PTZ cameras with precision using a game controller,
virtual controls, and custom command mapping. It is designed for camera
operators, live production teams, and AV users who want smoother motion,
fine control, and easy access to advanced VISCA commands. Written with the help of chatgpt 5.2.

Download latest release from here:
https://github.com/sek0002/PTZoptics_JS/releases/

------------------------------------------------------------------------

## What This App Does

PTZoptics_JS lets you control your PTZ camera in multiple ways:

-   Use a game controller for smooth pan/tilt/zoom movement
-   Fine-tune joystick sensitivity with a visual response curve editor
-   Send VISCA commands directly (presets, power, focus, exposure,
    etc.)
-   Use an on-screen virtual controller if no hardware controller is
    available
-   Create custom mappings for buttons and axes

This makes it ideal for live streaming, churches, studios, classrooms,
and event production.

------------------------------------------------------------------------

## Key Features

### Smooth Camera Motion

-   Adjustable deadzone for each axis
-   Expo (sensitivity curve) for precision at low speeds
-   Speed limiting and shaping
-   Axis inversion
-   Trigger-style zero-start mode
-   Visual live curve editor

### Game Controller Support

-   Map controller axes to:
    -   Pan
    -   Tilt
    -   Zoom
    -   Focus
-   Customize response for each axis independently
-   Speed bin mapping for predictable movement

### Virtual Controller (No Hardware Needed)

-   On-screen D-pad
-   A / B / X / Y buttons
-   Assign any VISCA command to each button
-   Instant execution when pressed

### Built-in VISCA Commands

Common camera operations included: - Power On / Off - Zoom In / Out /
Stop - Focus Auto / Manual - Pan / Tilt movement - White balance modes -
Exposure modes - Preset Save / Recall - Camera status queries

### Hex Command Editor

-   Create and edit your own VISCA commands
-   Send raw hex payloads
-   Useful for:
    -   Advanced camera features
    -   Testing
    -   Unsupported commands

### Future features underway 

-   performance and user interface optimisation
-   stability works
-   additional query functions
-   additional in-app image and presets functions
-   live video streaming within app
-   position based control for smoother pan/tilt/zoom
-   bug fixes
------------------------------------------------------------------------

If you use PTZOptics or other VISCA-compatible cameras, this tool gives
you more control than basic control panels.

------------------------------------------------------------------------

## Requirements

-   Python 3.9+
-   A VISCA-compatible PTZ camera
-   Network connection to the camera

Optional: - Game controller (Xbox/PlayStation/USB joystick)

------------------------------------------------------------------------

## Installation

### 1) Clone the repository

git clone https://github.com/yourname/PTZoptics_JS.git\
cd PTZoptics_JS

### 2) Install dependencies

pip install PySide6 pygame

Optional modules (recommended for full experience):

pip install PySide6-QtCharts

### 3) Run the app

python main1.py

------------------------------------------------------------------------

## Open License

This project is released under the MIT License.
