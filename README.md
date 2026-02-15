# PTZoptics_JS <img align="right" width="150" height="150" alt="UVCC" src="https://github.com/user-attachments/assets/8957cf51-e8a5-4ed3-80c0-2d139271ee60" />


**PTZoptics_JS** is a desktop application for controlling
VISCA-compatible PTZ cameras with precision using a game controller,
virtual controls, and custom command mapping. It is designed for camera
operators, live production teams, and AV users who want smoother motion,
fine control, and easy access to advanced VISCA commands. Written with the help of chatgpt 5.2.

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

### Clean Dark Interface

-   Always-dark theme
-   Live feedback display
-   Real-time movement visualization

------------------------------------------------------------------------

## Who This Is For

-   Church AV teams
-   Live stream operators
-   Video production crews
-   PTZ camera owners
-   Integrators and technicians

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
