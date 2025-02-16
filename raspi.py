from flask import Flask, Response, request, jsonify
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
from libcamera import controls
import threading
import RPi.GPIO as GPIO
import time

# Initialize Flask app
app = Flask(__name__)

# Initialize camera
picam2 = Picamera2()

# Configure camera with proper color controls
config = picam2.create_video_configuration(
    main={"size": (640, 480)},
    controls={
        "AwbMode": controls.AwbModeEnum.Auto,  # Auto white balance
        "AeEnable": True,                      # Auto exposure
        "Brightness": 0.0,                     # Neutral brightness
        "Contrast": 1.0                        # Default contrast
    }
)
picam2.configure(config)
picam2.start()

# Global variables and locks
last_command = ""
command_lock = threading.Lock()
detected_crop = ""
crop_lock = threading.Lock()

# Crop moisture and humidity data (example values)
CROP_MOISTURE = {
    "jute": {"humidity": "60-70%", "soil moisture": "50-60%"},
    "maize": {"humidity": "50-60%", "soil moisture": "40-50%"},
    "rice": {"humidity": "70-80%", "soil moisture": "60-70%"},
    "sugarcane": {"humidity": "50-60%", "soil moisture": "40-50%"},
    "wheat": {"humidity": "40-50%", "soil moisture": "30-40%"}
}

# Motor control setup
GPIO.setmode(GPIO.BCM)
IN1, IN2, IN3, IN4 = 17, 18, 22, 23  # Motor control pins
ENA, ENB = 25, 24  # Enable pins for PWM speed control

# Set pins as output
for pin in [IN1, IN2, IN3, IN4, ENA, ENB]:
    GPIO.setup(pin, GPIO.OUT)

# Enable PWM for speed control
pwmA = GPIO.PWM(ENA, 1000)  # 1000 Hz frequency
pwmB = GPIO.PWM(ENB, 1000)
pwmA.start(50)  # 50% duty cycle
pwmB.start(50)

# Motor control functions
def forward():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def backward():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)

def turn_left():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def turn_right():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)

def stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

# Endpoint to receive commands
@app.route('/command', methods=['POST'])
def receive_command():
    global last_command
    data = request.json
    command = data.get("command", "")
    with command_lock:
        last_command = command

    # Execute motor control based on command
    if command == "Moving Forward":
        forward()
    elif command == "Moving Backward":
        backward()
    elif command == "Turning Left":
        turn_left()
    elif command == "Turning Right":
        turn_right()
    elif command == "Lowering Sensor":
        stop()

    return {"status": "success", "received": command}, 200

# Load the TFLite model for crop classification
interpreter = tflite.Interpreter(model_path="/home/makeuoft/Downloads/crop_classifier.tflite")
interpreter.allocate_tensors()

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels (make sure this order matches your training)
class_names = ["jute", "maize", "rice", "sugarcane", "wheat"]

def preprocess_image(image):
    # Resize the image to match model input dimensions (150x150)
    image = cv2.resize(image, (150, 150))
    # Expand dimensions to add the batch size and normalize pixel values
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32) / 255.0
    return image

def predict_crop(image):
    processed = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], processed)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = class_names[np.argmax(predictions)]
    return predicted_label

def generate_frames():
    global last_command, detected_crop
    while True:
        # Capture a frame from the camera
        frame = picam2.capture_array()
        # Convert from RGB (Picamera2 default) to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Get crop classification for the current frame
        crop_type = predict_crop(frame)
        with crop_lock:
            detected_crop = crop_type  # Update the global detected crop
        
        # Overlay crop type on the frame
        cv2.putText(frame, f"Crop: {crop_type}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Get and overlay the last received command (if any)
        with command_lock:
            cmd = last_command
        if cmd:
            cv2.putText(frame, f"Command: {cmd}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/crop_data')
def get_crop_data():
    global detected_crop
    with crop_lock:
        crop = detected_crop
    if crop in CROP_MOISTURE:
        return jsonify({
            "crop": crop,
            "humidity": CROP_MOISTURE[crop]["humidity"],
            "soil_moisture": CROP_MOISTURE[crop]["soil moisture"]
        })
    else:
        return jsonify({"error": "No crop detected"}), 404

@app.route('/')
def index():
    return """
    <html>
      <head>
        <title>Pi Camera Stream</title>
      </head>
      <body>
        <h1>Pi Camera Live Stream</h1>
        <img src="/video_feed" style="width:640px">
      </body>
    </html>
    """

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        # Clean up GPIO on exit
        GPIO.cleanup()