import cv2
import mediapipe as mp
import numpy as np
import time
import RPi.GPIO as GPIO

# Initialize Mediapipe Face Mesh for facial landmarks detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Setup GPIO for motor control
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motor control pins
IN1, IN2 = 17, 27  # Horizontal motor
IN3, IN4 = 22, 23  # Vertical motor
IN5, IN6 = 5, 6    # Tilt motor
motor_pins = [IN1, IN2, IN3, IN4, IN5, IN6]

# Set all motor pins as output
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Motor control functions
def motor_forward(in1, in2, duration=0.5):
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)
    time.sleep(duration)
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)

def motor_reverse(in1, in2, duration=0.5):
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)

# Camera setup
camera = cv2.VideoCapture(0)
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Flag to track automation state
automation_running = False

# Function to count fingers
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

# Function to calculate and control motors
def calculate_face_orientation_and_dof(landmarks):
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

    if automation_running:
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

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers_up = count_fingers(hand_landmarks.landmark)
            if fingers_up == 5:
                automation_running = True
                print("Automation Started")
            elif fingers_up == 0:
                automation_running = False
                print("Automation Stopped")

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            calculate_face_orientation_and_dof(face_landmarks.landmark)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
GPIO.cleanup()
