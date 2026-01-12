from flask import Flask, render_template, Response, jsonify
import cv2
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import time

app = Flask(__name__)

# --- Load Model & Assets ---
model = load_model('model_final.h5')
class_dictionary = {0: 'no_car', 1: 'car'}

# Video Capture
video_path = 'car_test.mp4'
cap = cv2.VideoCapture(video_path)

# Load Positions
with open('carposition.pkl', 'rb') as f:
    positionList = pickle.load(f)

# Dimensions for slanted crop
width = 70
height = 90

# --- Global State for Periodic Detection ---
last_parking_status = []
last_space_count = 0
start_time = None

def adjust_brightness(img, threshold=80, target=120):
    """Adjust brightness if image is too dark"""
    brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img)
    if brightness < threshold:
        factor = target / brightness
        return cv2.convertScaleAbs(img, alpha=factor, beta=0)
    return img

def run_detection(img):
    """Runs model prediction and updates global state. Only call periodically."""
    global last_parking_status, last_space_count
    
    imgCrops = []
    spaceCounter = 0
    parking_status_list = [] 
    
    for pos in positionList:
        x, y = pos
        offset = width // 4
        cropped_img = img[y:y+height, x:x+width+offset]
        
        if cropped_img.shape[0] < height or cropped_img.shape[1] < (width+offset):
            continue

        cropped_img = adjust_brightness(cropped_img)

        # Slanted Masking
        mask = np.zeros((height, width + offset), dtype=np.uint8)
        points = np.array([
            [0, 0], [width, 0], [width + offset, height], [offset, height]
        ], np.int32)
        cv2.fillPoly(mask, [points], 255)
        res = cv2.bitwise_and(cropped_img, cropped_img, mask=mask)

        imgResized = cv2.resize(res, (48,48))
        imgNormalized = imgResized/255.0
        imgCrops.append(imgNormalized)
    
    # Batch predict
    if len(imgCrops) > 0:
        imgCrops = np.array(imgCrops)
        predictions = model.predict(imgCrops)

        for i, pos in enumerate(positionList):
            if i < len(predictions):
                intId = np.argmax(predictions[i])
                label = class_dictionary[intId]
                if label == 'no_car':
                    spaceCounter += 1
                
                parking_status_list.append({
                    "id": i,
                    "status": int(intId)
                })
        
        last_parking_status = parking_status_list
        last_space_count = spaceCounter

def draw_overlay(img):
    """Draws boxes based on the last known parking status"""
    for i, item in enumerate(last_parking_status):
        if i >= len(positionList): break
        pos = positionList[i]
        x, y = pos
        label = class_dictionary[item['status']]
        
        if label == 'no_car':
            color = (0, 255, 0)
            thickness = 2
            textColor = (0, 0, 0)
        else:
            color = (0, 0, 255)
            thickness = 2
            textColor = (255, 255, 255)

        # Slanted box points
        offset = width // 4
        points = np.array([
            [pos[0], pos[1]],
            [pos[0] + width, pos[1]],
            [pos[0] + width + offset, pos[1] + height],
            [pos[0] + offset, pos[1] + height]
        ], np.int32)
        
        cv2.polylines(img, [points], isClosed=True, color=color, thickness=thickness)
        
        # Draw Labels
        textSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        textX = x
        textY = y + height - 5
        cv2.rectangle(img, (textX, textY - textSize[1] - 5), (textX + textSize[0] + 60, textY + 2), color, -1)
        display_text = f"ID:{i} {label}"
        cv2.putText(img, display_text, (textX + 3, textY - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, textColor, 1)

    cv2.putText(img, f'Space Count: {last_space_count}', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img

def generate_frames():
    global start_time
    
    # Reset/Start timer
    start_time = time.time()
    
    while True:
        # Time-based synchronization logic from testing4.py
        elapsed_real_ms = (time.time() - start_time) * 1000
        cap.set(cv2.CAP_PROP_POS_MSEC, elapsed_real_ms)
        
        success, img = cap.read()
        if not success:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            start_time = time.time()
            continue

        img = cv2.resize(img, (1280, 720))
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Run detection every 30 frames
        if current_frame % 30 == 0: 
             run_detection(img)
        
        # Always draw overlay on every frame using last status
        if last_parking_status:
            img = draw_overlay(img)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/space_count')
def space_count():
    # Return count from global state (updated by the video thread)
    occupied = len(positionList) - last_space_count
    return jsonify(free=last_space_count, occupied=occupied)

if __name__ == "__main__":
    app.run(debug=True)
