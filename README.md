# Tello-obstacle-avoidance-and-target-tracking-with-YOLOv8

This repository contains code for controlling a Tello drone and using YOLO models for object detection. The drone identifies and navigates towards a specific target using a YOLOv8 model, while avoiding obstacles in its path. This project leverages the capabilities of YOLOv8 and custom-trained YOLOv5 models.

## Requirements

- Python 3.x
- djitellopy
- OpenCV
- PyTorch
- Ultralytics YOLO

## Installation

1. Install the required libraries:

```bash
pip install djitellopy opencv-python torch ultralytics
```

2. Clone this repository:

```bash
git clone <repository_url>
```

3. Download the YOLO models and place them in the `models` directory:
   - YOLOv8: [Ultralytics YOLOv8](https://ultralytics.com/yolov8)
   - Custom YOLOv5 model (trained by us)

## Usage

1. Ensure the Tello drone is connected and ready to fly.
2. Run the main script:

```bash
python main.py
```

The script will start the Tello drone, turn on the video stream, and begin object detection. The drone will navigate towards the detected target while avoiding obstacles.

## Code Overview

The main components of the code include:

- **Model Initialization**: Loads the YOLOv8 model for detection and a custom-trained YOLOv5 model for specific target detection.
- **Drone Initialization**: Connects to the Tello drone, starts the video stream, and checks the battery level.
- **Object Detection and Navigation**: Processes video frames to detect objects, calculates errors for navigation, and sends commands to the drone to move towards the target while avoiding obstacles.

## Contributing

You can contribute to this project by finding ways to improve the drone's navigation in a room. Feel free to fork this repository, make your changes, and submit a pull request.

## Acknowledgments

- Special thanks to [Ultralytics](https://ultralytics.com) for their YOLOv8 model, which significantly enhanced this project's object detection capabilities.

## Additional Resources

For more information on operating the Tello drone, refer to [Tello Drone Help Book](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20User%20Manual%20v1.4.pdf).

## Notes

- The YOLOv5 model is custom-trained by us to detect specific targets not present in the standard YOLOv8 model database.
- The code was modified to use a bottle from the YOLOv8 model as the target. Running both YOLOv5 and YOLOv8 models simultaneously causes lag, so we opted for this approach to ensure smooth operation.

---

Feel free to reach out if you have any questions or need further assistance.

Happy flying!
