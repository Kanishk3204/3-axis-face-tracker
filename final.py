import RPi.GPIO as GPIO
import cv2
import numpy as np
import time

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motor pins
IN1, IN2 = 17, 27  # Motor 1 (Horizontal)
IN3, IN4 = 22, 23  # Motor 2 (Vertical)
IN5, IN6 = 24, 25  # Motor 3 (Tilt)

motor_pins = [IN1, IN2, IN3, IN4, IN5, IN6]

for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Motor control functions
def motor_forward(in1, in2, duration):
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)
    time.sleep(duration)
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)

def motor_reverse(in1, in2, duration):
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)

# DNN Face Detection Setup
prototxt_path = "deploy.prototxt.txt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
face_detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Webcam initialization
cap = cv2.VideoCapture(0)
movement_threshold = 50
screen_center_x, screen_center_y = 0, 0

# Tilt calculation function
def calculate_tilt(face_box, frame):
    (startX, startY, endX, endY) = face_box
    face_roi = frame[startY:endY, startX:endX]
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    face_width = endX - startX
    left_half = gray_face[:, :face_width // 2]
    right_half = gray_face[:, face_width // 2:]
    left_brightness = np.mean(left_half)
    right_brightness = np.mean(right_half)

    if left_brightness > right_brightness + 10:
        return "Tilted Left"
    elif right_brightness > left_brightness + 10:
        return "Tilted Right"
    else:
        return "Neutral"

# Movement detection function
def detect_movement(face_box):
    global screen_center_x, screen_center_y
    (startX, startY, endX, endY) = face_box
    face_center_x = (startX + endX) // 2
    face_center_y = (startY + endY) // 2

    if face_center_x < screen_center_x - movement_threshold:
        horizontal = "Left"
    elif face_center_x > screen_center_x + movement_threshold:
        horizontal = "Right"
    else:
        horizontal = "Center"

    if face_center_y < screen_center_y - movement_threshold:
        vertical = "Up"
    elif face_center_y > screen_center_y + movement_threshold:
        vertical = "Down"
    else:
        vertical = "Center"

    return horizontal, vertical

# Initialize screen center
ret, frame = cap.read()
if ret:
    screen_center_x = frame.shape[1] // 2
    screen_center_y = frame.shape[0] // 2

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video.")
            break

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_detector.setInput(blob)
        detections = face_detector.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                # Detect tilt and movement
                tilt_status = calculate_tilt((startX, startY, endX, endY), frame)
                horizontal, vertical = detect_movement((startX, startY, endX, endY))

                # Control motors based on movement
                if horizontal == "Left":
                    motor_forward(IN1, IN2, 0.1)
                elif horizontal == "Right":
                    motor_reverse(IN1, IN2, 0.1)

                if vertical == "Up":
                    motor_forward(IN3, IN4, 0.1)
                elif vertical == "Down":
                    motor_reverse(IN3, IN4, 0.1)

                if tilt_status == "Tilted Left":
                    motor_forward(IN5, IN6, 0.1)
                elif tilt_status == "Tilted Right":
                    motor_reverse(IN5, IN6, 0.1)

                # Display status
                cv2.putText(frame, f"Tilt: {tilt_status}", (startX, startY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Horizontal: {horizontal}", (startX, startY - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Vertical: {vertical}", (startX, startY - 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                print(f"Movement: Horizontal: {horizontal}, Vertical: {vertical}, Tilt: {tilt_status}")

        # Show the frame
        cv2.imshow("Face Detection and Tracking", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Exiting program.")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
