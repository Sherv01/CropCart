# Smart Farming Car with Crop Detection

![Project Demo](demo.gif) <!-- Add a GIF or image of your project in action -->

## Overview
This project is a **smart farming car** that uses **machine learning** to detect crops and provide real-time recommendations for humidity and soil moisture. The system integrates a Raspberry Pi, a camera, and motors, controlled via a **Streamlit dashboard**. It’s designed to automate crop monitoring and optimize resource usage for precision agriculture.

---

## Features
- **Real-Time Crop Detection**:  
  - Uses a TensorFlow Lite model to classify crops (e.g., jute, maize, rice).  
  - Overlays crop type on the live camera feed.  

- **Streamlit Dashboard**:  
  - Interactive interface for controlling the car (forward, backward, left, right, stop).  
  - Displays live camera feed and crop-specific recommendations.  

- **Precision Agriculture**:  
  - Provides crop-specific humidity and soil moisture recommendations.  
  - Optimizes resource usage for sustainable farming.  

---

## How It Works
1. **Raspberry Pi**:  
   - Captures live video feed using a Pi camera.  
   - Runs a Flask server to handle crop detection and motor control.  

2. **Machine Learning**:  
   - TensorFlow Lite model processes frames to detect crops.  
   - Sends crop data and recommendations to the Streamlit dashboard.  

3. **Streamlit Dashboard**:  
   - Provides a user-friendly interface for controlling the car and viewing real-time data.  

---

## Setup Instructions

### Prerequisites
- Raspberry Pi with Raspberry Pi OS.
- Pi Camera module.
- Motor driver and motors.
- Python 3.7+.

### Installation
1. Clone the repository.
2. Set up a virtual environment.
3. Install dependencies
4. Connect the hardware:
   - Attach the Pi Camera to the Raspberry Pi.
   - Connect the motor driver and motors to the GPIO pins as specified in the code.
5. Run the Flask server on the Raspberry Pi using `python stream.py`.
6. Run the Streamlit app on your local machine or server using `streamlit run app.py`.

---

## Technologies Used
- **Hardware**: Raspberry Pi, Pi Camera, Motor Driver, Motors.  
- **Software**: Flask, Streamlit, TensorFlow Lite, OpenCV.  
- **Languages**: Python.  

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- Inspired by precision agriculture and IoT projects.  
- Built with the help of the Raspberry Pi and TensorFlow Lite communities.  

---

Feel free to contribute or reach out with questions! 🚀
