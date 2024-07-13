import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# Initialize volume control variables
volume = 0
volume_scale = 100
volume_max = 100
volume_min = 0

while True:
    ret, frame = cap.read()

    # Get the frame dimensions
    image_height, image_width, _ = frame.shape

    # Detect hands and draw landmarks
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calculate the distance between the thumb and the index finger landmarks
            thumb_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_x = int(thumb_landmark.x * image_width)
            thumb_y = int(thumb_landmark.y * image_height)
            index_x = int(index_finger_landmark.x * image_width)
            index_y = int(index_finger_landmark.y * image_height)
            finger_distance = math.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

            # Map the finger distance to the volume scale
            volume = np.interp(finger_distance, [0, image_width], [volume_min, volume_max])
            volume = int(volume)

            # Set the system volume using pyautogui
            pyautogui.press('volumedown') if volume < volume_scale else pyautogui.press('volumeup')

    # Display the volume level
    cv2.putText(frame, f"Volume: {volume}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Volume Gesture Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
