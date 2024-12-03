import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize Mediapipe Face Mesh for facial landmarks detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Camera setup
camera = cv2.VideoCapture(0)

# Get camera frame size
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Timer setup for 5-second intervals
last_print_time = time.time()

# move up,down move left,right tilt left,right
target_binary_array = [
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

actual_results = []  # To store actual binary results for each frame

# Function to calculate yaw, pitch, roll, and DOF movement
def calculate_face_orientation_and_dof(landmarks):
    # Get coordinates for key landmarks (nose tip, left eye, right eye, chin)
    # nose_tip = np.array([landmarks[1].x * frame_width, landmarks[1].y * frame_height])  # Nose tip (landmark 1)
    left_eye = np.array([landmarks[33].x * frame_width, landmarks[33].y * frame_height])  # Left eye (landmark 33)
    right_eye = np.array([landmarks[263].x * frame_width, landmarks[263].y * frame_height])  # Right eye (landmark 263)
    chin = np.array([landmarks[152].x * frame_width, landmarks[152].y * frame_height])

    # Calculate the midpoints for face center calculations
    eye_midpoint = (left_eye + right_eye) / 2
    vertical_midpoint = (eye_midpoint + chin) / 2

    # Calculate pitch based on the difference in Y coordinates
    # pitch_angle = nose_tip[1] - vertical_midpoint[1]  # Positive if nose is above eye midpoint, negative if below

    # Calculate the tilt angle (roll) between the eyes
    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    roll_angle = np.degrees(np.arctan2(delta_y, delta_x))


    # Calculate yaw (left/right rotation): angle between nose and vertical midpoint
    # yaw_angle = np.degrees(np.arctan2(nose_tip[0] - vertical_midpoint[0], frame_width))

    # Calculate horizontal and vertical displacement from frame center
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2
    horizontal_displacement = vertical_midpoint[0] - frame_center_x
    vertical_displacement = vertical_midpoint[1] - frame_center_y

    # Print the yaw, pitch, roll, and displacement values every 2 seconds
    # print(f"Yaw: {yaw_angle:.2f}°")
    # print(f"Pitch: {pitch_angle:.2f}°")
    print(f"Roll: {roll_angle:.2f}°")
    print(f"Horizontal Displacement: {horizontal_displacement}px")
    print(f"Vertical Displacement: {vertical_displacement}px")

    # Determine binary movement in 6 DOF based on threshold values
    move_left_right = "Move Left" if horizontal_displacement > 40 else "Move Right" if horizontal_displacement < -40 else "Center"
    move_up_down = "Move Down" if vertical_displacement > 50 else "Move Up" if vertical_displacement < -50 else "Center"
    tilt_correction = "Tilt Left" if roll_angle > 10 else "Tilt Right" if roll_angle < -10 else "Straight"

    actual_binary = [
        1 if vertical_displacement < -50 else 0,  # Move Up
        1 if vertical_displacement > 50 else 0,  # Move Down
        1 if horizontal_displacement > 40 else 0,  # Move Left
        1 if horizontal_displacement < -40 else 0,  # Move Right
        1 if roll_angle > 10 else 0,  # Tilt Left
        1 if roll_angle < -10 else 0  # Tilt Right
    ]

    actual_results.append(actual_binary)

    # Print movement suggestions
    print(f"Horizontal Movement: {move_left_right}")
    print(f"Vertical Movement: {move_up_down}")
    print(f"Tilt Correction: {tilt_correction}")
    print(actual_binary)
    print("---")
cnt =0
while True:
    # Capture frame-by-frame from the camera
    ret, frame = camera.read()

    if not ret:
        print("Failed to capture image")
        break

    # Convert frame to RGB for Mediapipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect facial landmarks using Face Mesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract landmark positions (list of normalized coordinates)
            landmarks = face_landmarks.landmark

            # Check if 2 seconds have passed
            current_time = time.time()
            if current_time - last_print_time >= 5:
                # Calculate and print face orientation and movement every 2 seconds
                calculate_face_orientation_and_dof(landmarks)
                last_print_time = current_time
                cnt+=1

    # Display the resulting frame without drawing landmarks
    cv2.imshow('Face Tracker with Yaw, Pitch, Roll', frame)
    # Add a small delay to prevent freezing (UI responsiveness)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Break loop on 'q' key press
    if cnt==len(target_binary_array):
        break

correct_predictions = np.sum(np.all(np.array(actual_results[-len(target_binary_array):]) == np.array(target_binary_array), axis=1))
accuracy = (correct_predictions / len(target_binary_array)) * 100
print(f"Accuracy: {accuracy:.2f}%")
# Cleanup
camera.release()
cv2.destroyAllWindows()
