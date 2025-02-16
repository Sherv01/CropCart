from flask import Flask, Response, request
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
from libcamera import controls
import threading

app = Flask(__name__)
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

# Global variable and lock for storing the last received command
last_command = ""
command_lock = threading.Lock()

@app.route('/command', methods=['POST'])
def receive_command():
    global last_command
    data = request.json
    with command_lock:
        last_command = data.get("command", "")
    return {"status": "success", "received": last_command}, 200

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
    global last_command
    while True:
        # Capture a frame from the camera
        frame = picam2.capture_array()
        # Convert from RGB (Picamera2 default) to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Get crop classification for the current frame
        crop_type = predict_crop(frame)
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
    app.run(host='0.0.0.0', port=5000, debug=False)
