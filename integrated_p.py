import cv2
import mediapipe as mp
import numpy as np
import time
import RPi.GPIO as GPIO
import threading

# Initialize MediaPipe Modules
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motor pins
IN1, IN2 = 22, 23   # Horizontal
IN3, IN4 = 24, 25   # Vertical
IN5, IN6 = 17, 27   # Tilt
motor_pins = [IN1, IN2, IN3, IN4, IN5, IN6]

# LED pins
LED_NEAR, LED_IDEAL, LED_FAR = 5, 6, 13
led_pins = [LED_NEAR, LED_IDEAL, LED_FAR]

# Set all motor and LED pins as output
for pin in motor_pins + led_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Constants
KNOWN_FACE_WIDTH = 14  # cm
FOCAL_LENGTH = 380  # Adjusted via calibration
automation_running = False
frame = None  # Shared frame variable
lock = threading.Lock()  # Lock for safe access to shared frame

# Motor control functions
def motor_forward(in1, in2, duration=0.035):
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)
    time.sleep(duration)
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)

def motor_reverse(in1, in2, duration=0.035):
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)

# Function to estimate distance
def estimate_distance(face_width):
    return (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / face_width

# LED control function
def control_led(distance):
    GPIO.output(LED_NEAR, GPIO.LOW)
    GPIO.output(LED_IDEAL, GPIO.LOW)
    GPIO.output(LED_FAR, GPIO.LOW)

    if distance < 25:
        GPIO.output(LED_NEAR, GPIO.HIGH)
        return "Near"
    elif distance > 60:
        GPIO.output(LED_FAR, GPIO.HIGH)
        return "Far"
    else:
        GPIO.output(LED_IDEAL, GPIO.HIGH)
        return "Ideal"

# Gesture recognition function
def count_fingers(hand_landmarks):
    fingers = [
        1 if hand_landmarks[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks[mp_hands.HandLandmark.THUMB_IP].x else 0
    ]
    for tip_id, dip_id in [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_DIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_DIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_DIP)
    ]:
        fingers.append(1 if hand_landmarks[tip_id].y < hand_landmarks[dip_id].y else 0)
    return fingers.count(1)

# Face orientation calculation
def calculate_face_orientation_and_dof(landmarks, frame_width, frame_height):
    left_eye = np.array([landmarks[33].x * frame_width, landmarks[33].y * frame_height])
    right_eye = np.array([landmarks[263].x * frame_width, landmarks[263].y * frame_height])
    chin = np.array([landmarks[152].x * frame_width, landmarks[152].y * frame_height])

    eye_midpoint = (left_eye + right_eye) / 2
    vertical_midpoint = (eye_midpoint + chin) / 2

    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    roll_angle = np.degrees(np.arctan2(delta_y, delta_x))

    frame_center_x, frame_center_y = frame_width // 2, frame_height // 2
    horizontal_displacement = vertical_midpoint[0] - frame_center_x
    vertical_displacement = vertical_midpoint[1] - frame_center_y

    return horizontal_displacement, vertical_displacement, roll_angle

# Camera setup
camera = cv2.VideoCapture(0)
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Thread functions
def process_face_orientation():
    global frame
    while True:
        with lock:
            if frame is None:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks and automation_running:
            for face_landmarks in results.multi_face_landmarks:
                horizontal_displacement, vertical_displacement, roll_angle = calculate_face_orientation_and_dof(
                    face_landmarks.landmark, frame_width, frame_height
                )

                if horizontal_displacement > 40:
                    motor_forward(IN1, IN2)
                elif horizontal_displacement < -40:
                    motor_reverse(IN1, IN2)

                if vertical_displacement > 50:
                    motor_forward(IN3, IN4)
                elif vertical_displacement < -50:
                    motor_reverse(IN3, IN4)

                if roll_angle > 10:
                    motor_forward(IN5, IN6)
                elif roll_angle < -10:
                    motor_reverse(IN5, IN6)

def process_distance_measurement():
    global frame
    while True:
        with lock:
            if frame is None:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                face_width = bboxC.width * frame_width
                distance = estimate_distance(face_width)
                control_led(distance)

def process_gesture_recognition():
    global frame, automation_running
    while True:
        with lock:
            if frame is None:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(rgb_frame)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                fingers_up = count_fingers(hand_landmarks.landmark)
                if fingers_up == 5:
                    automation_running = True
                elif fingers_up == 0:
                    automation_running = False

# Start threads
t1 = threading.Thread(target=process_face_orientation, daemon=True)
t2 = threading.Thread(target=process_distance_measurement, daemon=True)
t3 = threading.Thread(target=process_gesture_recognition, daemon=True)

t1.start()
t2.start()
t3.start()

# Main loop to update frames
while True:
    ret, frame = camera.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
GPIO.cleanup()
