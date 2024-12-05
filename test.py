import cv2
import numpy as np

# Load pre-trained DNN face detector
prototxt_path = "deploy.prototxt.txt"  # Path to deploy.prototxt file
model_path = "res10_300x300_ssd_iter_140000.caffemodel"  # Path to caffemodel file
face_detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Movement threshold
movement_threshold = 50  # For detecting significant movements

# Screen center (default values)
screen_center_x = 0
screen_center_y = 0

def calculate_tilt(face_box, frame):
    """
    Estimate head tilt based on the face bounding box.
    """
    (startX, startY, endX, endY) = face_box
    face_width = endX - startX
    face_height = endY - startY

    # Define regions of interest for tilt estimation (focus on the nose/midface)
    face_roi = frame[startY:endY, startX:endX]
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # Split face vertically into left and right halves
    left_half = gray_face[:, :face_width // 2]
    right_half = gray_face[:, face_width // 2:]

    # Calculate average pixel intensity for each half (brightness comparison)
    left_brightness = np.mean(left_half)
    right_brightness = np.mean(right_half)

    # Determine tilt direction based on brightness difference
    if left_brightness > right_brightness + 10:
        return "Tilted Left"
    elif right_brightness > left_brightness + 10:
        return "Tilted Right"
    else:
        return "Neutral"

def detect_movement(face_box):
    """
    Detects up, down, left, and right movement based on the face position.
    """
    global screen_center_x, screen_center_y
    (startX, startY, endX, endY) = face_box

    # Calculate face center
    face_center_x = (startX + endX) // 2
    face_center_y = (startY + endY) // 2

    # Determine direction based on movement thresholds
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

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video.")
        break

    # Prepare frame for DNN processing
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Process faces with confidence > 50%
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw face bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Detect tilt and movement
            tilt_status = calculate_tilt((startX, startY, endX, endY), frame)
            horizontal, vertical = detect_movement((startX, startY, endX, endY))

            # Display movement and tilt status
            cv2.putText(frame, f"Tilt: {tilt_status}", (startX, startY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Horizontal: {horizontal}", (startX, startY - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Vertical: {vertical}", (startX, startY - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Print movement details
            print(f"Movement: Horizontal: {horizontal}, Vertical: {vertical}, Tilt: {tilt_status}")

    # Show the frame
    cv2.imshow("Face Detection with Tilt and Movement", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
