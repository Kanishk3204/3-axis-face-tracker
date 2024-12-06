import cv2
import numpy as np
import math
import RPi.GPIO as GPIO
import time

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motor pins
IN1, IN2 = 22, 23  # Motor 1 (Horizontal)
IN3, IN4 = 24, 25  # Motor 2 (Vertical)
IN5, IN6 = 17, 27  # Motor 3 (Tilt)
motor_pins = [IN1, IN2, IN3, IN4, IN5, IN6]

# Initialize GPIO pins
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Motor control functions
def motor_forward(in1, in2, duration=0.2):
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)
    time.sleep(duration)
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)

def motor_reverse(in1, in2, duration=0.2):
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)

# Load pre-trained DNN face detector
prototxt_path = "deploy.prototxt.txt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
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
            tilt_status = calculate_tilt(eyes)
            horizontal, vertical = detect_movement((startX, startY, endX, endY))

            # Motor control logic
            if horizontal == "Left":
                motor_forward(IN1, IN2)
            elif horizontal == "Right":
                motor_reverse(IN1, IN2)
            if vertical == "Up":
                motor_forward(IN3, IN4)
            elif vertical == "Down":
                motor_reverse(IN3, IN4)
            if tilt_status == "Tilted Left":
                motor_forward(IN5, IN6)
            elif tilt_status == "Tilted Right":
                motor_reverse(IN5, IN6)

            cv2.putText(frame, f"Tilt: {tilt_status}", (startX, startY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Horizontal: {horizontal}", (startX, startY - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Vertical: {vertical}", (startX, startY - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Face Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
