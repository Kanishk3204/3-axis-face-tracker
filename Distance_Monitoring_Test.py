import cv2
import mediapipe as mp
import numpy as np
import threading

# Constants
KNOWN_FACE_WIDTH = 14  # cm
FOCAL_LENGTH = 380  # Adjust based on calibration

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Store actual & estimated distances
actual_distances = []
estimated_distances = []
user_input = None  # Global variable to store input

def estimate_distance(face_width):
    """Calculate estimated distance based on face width in pixels."""
    return (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / face_width

def get_user_input():
    """Run input in a separate thread to prevent blocking."""
    global user_input
    while True:
        user_input = input("\nEnter actual distance (cm) or 'q' to quit: ")
        if user_input.lower() == 'q':
            break

# Start input thread
input_thread = threading.Thread(target=get_user_input, daemon=True)
input_thread.start()

# Open Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Reduce width
cap.set(4, 480)  # Reduce height

frame_count = 0  # Track frames to skip processing

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % 3 != 0:  # Skip frames for better speed
        continue
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            face_width = bboxC.width * w  

            # Estimate distance
            estimated_distance = estimate_distance(face_width)

            # Draw bounding box & label
            x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Est: {int(estimated_distance)} cm", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Face Distance Accuracy Test", frame)

    if user_input:
        if user_input.lower() == 'q':
            break
        try:
            actual_distance = float(user_input)
            actual_distances.append(actual_distance)
            estimated_distances.append(estimated_distance)
            print(f"✅ Recorded: Actual = {actual_distance} cm, Estimated = {round(estimated_distance, 2)} cm\n")
        except ValueError:
            print("⚠️ Invalid input. Please enter a numeric value.")
        user_input = None  # Reset input

cap.release()
cv2.destroyAllWindows()

# Compute accuracy metrics
if actual_distances and estimated_distances:
    actual_distances = np.array(actual_distances)
    estimated_distances = np.array(estimated_distances)

    mae = np.mean(np.abs(actual_distances - estimated_distances))
    mpe = np.mean(np.abs((actual_distances - estimated_distances) / actual_distances) * 100)

    print("\n📊 Accuracy Metrics:")
    print(f"🔹 Mean Absolute Error (MAE): {round(mae, 2)} cm")
    print(f"🔹 Mean Percentage Error (MPE): {round(mpe, 2)}%")
    print("\n✅ Testing Completed.")
else:
    print("\n⚠️ No data recorded. Please run the test again.")
