# This is main test python file for Parking test 1

import cv2
import pickle
import numpy as np
import json
from tensorflow.keras.models import load_model

model = load_model('model_final.h5')

class_dictionary = {0: 'no_car', 1:'car'}

video = cv2.VideoCapture('car_test.mp4')
import time
start_time = time.time()

with open('carposition.pkl', 'rb') as f:
    positionList = pickle.load(f)

# Optimization Globals
last_parking_status = []
last_space_count = 0
frame_counter = 0

width = 70
height = 90

# -----------------------------------------------------------------------------------

def adjust_brightness(img, threshold=80, target=120):
    """Adjust brightness if image is too dark"""
    brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img)
    if brightness < threshold:
        factor = target / brightness
        return cv2.convertScaleAbs(img, alpha=factor, beta=0)
    return img

def checkingCarParking(img):
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
    
    imgCrops = np.array(imgCrops)
    predictions = model.predict(imgCrops)

    for i, pos in enumerate(positionList):
        intId = np.argmax(predictions[i])
        label = class_dictionary[intId]
        if label == 'no_car':
            spaceCounter += 1
        
        parking_status_list.append({
            "id": i,
            "status": int(intId)
        })
    
    # Store results for frames where we don't run model
    last_parking_status = parking_status_list
    last_space_count = spaceCounter
    
    with open('parking_status.json', 'w') as f:
        json.dump(parking_status_list, f, indent=4)

def drawParkingStatus(img, status_list, space_count):
    """Draws the last known predictions on the current frame"""
    for i, item in enumerate(status_list):
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
        
        textSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        textX = x
        textY = y + height - 5
        cv2.rectangle(img, (textX, textY - textSize[1] - 5), (textX + textSize[0] + 60, textY + 2), color, -1)
        display_text = f"ID:{i} {label}"
        cv2.putText(img, display_text, (textX + 3, textY - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, textColor, 1)
    
    cv2.putText(img, f'Space Count: {space_count}', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

while True:
    elapsed_real_ms = (time.time() - start_time) * 1000
    video.set(cv2.CAP_PROP_POS_MSEC, elapsed_real_ms)
    
    ret, image = video.read()
    #if not ret:
    #    # Loop the video
    #    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #    start_time = time.time()
    #    continue

    # Update frame_counter based on the actual frame position for periodic detection
    current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    
    image = cv2.resize(image, (1280, 720))
    
    # Run prediction approximately every 30 frames
    if current_frame % 30 == 0:
        checkingCarParking(image)
    
    if last_parking_status:
        drawParkingStatus(image, last_parking_status, last_space_count)
    
    cv2.imshow("Image", image)
    
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()