import time
import cv2
import numpy as np
import math

# Load pre-trained DNN face detector
prototxt_path = r"C:\Users\Kashish Goel\Desktop\btp\deploy.prototxt.txt"
model_path = r"C:\Users\Kashish Goel\Desktop\btp\res10_300x300_ssd_iter_140000.caffemodel"

face_detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Load Haar cascade for eyes
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Movement threshold
movement_threshold = 50

# Initialize screen center
ret, frame = cap.read()
screen_center_x, screen_center_y = (frame.shape[1] // 2, frame.shape[0] // 2) if ret else (0, 0)

# Timer setup for 5-second intervals
last_print_time = time.time()

# Initialize ground truth (19 sets of 6 binary values)
#[left,right,up,down,tilt left,tilt right]
ground_truths = [
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 1],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0],
]

collected_predictions = []

def evaluate_accuracy(predictions, ground_truths):
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    # Extract roll (tilt correction), horizontal movement, and vertical movement
    roll_accuracy = np.mean(predictions[:, 4:6] == ground_truths[:, 4:6])
    horizontal_accuracy = np.mean(predictions[:, 0:2] == ground_truths[:, 0:2])
    vertical_accuracy = np.mean(predictions[:, 2:4] == ground_truths[:, 2:4])
    
    return roll_accuracy, horizontal_accuracy, vertical_accuracy

def calculate_tilt(eyes):
    if len(eyes) < 2:
        return "Neutral"
    eyes = sorted(eyes, key=lambda x: x[0])
    left_eye = (eyes[0][0] + eyes[0][2] // 2, eyes[0][1] + eyes[0][3] // 2)
    right_eye = (eyes[1][0] + eyes[1][2] // 2, eyes[1][1] + eyes[1][3] // 2)
    dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))
    return "Tilted Left" if angle > 5 else "Tilted Right" if angle < -5 else "Neutral"

def detect_movement(face_box):
    global screen_center_x, screen_center_y
    startX, startY, endX, endY = face_box
    face_center_x = (startX + endX) // 2
    face_center_y = (startY + endY) // 2
    horizontal = "Left" if face_center_x < screen_center_x - movement_threshold else \
                 "Right" if face_center_x > screen_center_x + movement_threshold else "Center"
    vertical = "Up" if face_center_y < screen_center_y - movement_threshold else \
               "Down" if face_center_y > screen_center_y + movement_threshold else "Center"
    return horizontal, vertical

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video.")
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                startX, startY, endX, endY = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                face_region = frame[startY:endY, startX:endX]
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                eyes = eye_cascade.detectMultiScale(gray_face)
                current_time = time.time()
                tilt_status= None
                horizontal = None
                vertical = None
                if current_time - last_print_time >= 5:
                    tilt_status = calculate_tilt(eyes)
                    horizontal, vertical = detect_movement((startX, startY, endX, endY))
                    last_print_time = current_time
                    predicted = [
                        1 if horizontal == "Left" else 0,
                        1 if horizontal == "Right" else 0,
                        1 if vertical == "Up" else 0,
                        1 if vertical == "Down" else 0,
                        1 if tilt_status == "Tilted Left" else 0,
                        1 if tilt_status == "Tilted Right" else 0
                    ]
                    
                    collected_predictions.append(predicted)
                    
                    if len(collected_predictions) >= 19:
                        roll_acc, hor_acc, ver_acc = evaluate_accuracy(collected_predictions, ground_truths)
                        print(f"Evaluation - Roll Accuracy: {roll_acc:.2f}, Horizontal Accuracy: {hor_acc:.2f}, Vertical Accuracy: {ver_acc:.2f}")
                        collected_predictions.clear()
                    print(f"Horizontal Movement: {horizontal}")
                    print(f"Vertical Movement: {vertical}")
                    print(f"Tilt Correction: {tilt_status}")
                    print("---")

    # Show the resulting frame
    cv2.imshow('Hand Gesture and Face Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
