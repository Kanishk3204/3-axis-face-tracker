import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Flag to track automation state
automation_running = False

# Function to count fingers
def count_fingers(hand_landmarks):
    fingers = []

    # Thumb
    if hand_landmarks[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks[mp_hands.HandLandmark.THUMB_IP].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Four fingers
    for tip_id, dip_id in [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_DIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_DIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_DIP)
    ]:
        if hand_landmarks[tip_id].y < hand_landmarks[dip_id].y:  # Check if the tip is above the DIP joint
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)


# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a natural interaction
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count fingers
            fingers_up = count_fingers(hand_landmarks.landmark)

            # Display the finger count
            cv2.putText(frame, f"Fingers: {fingers_up}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Start automation if all fingers are up (palm)
            if fingers_up == 5 and not automation_running:
                automation_running = True
                print("Automation Started")
                cv2.putText(frame, "Automation Started", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Stop automation if no fingers are up (fist)
            elif fingers_up == 0 and automation_running:
                automation_running = False
                print("Automation Stopped")
                cv2.putText(frame, "Automation Stopped", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Gesture Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
