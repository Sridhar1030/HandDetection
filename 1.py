import cv2
import mediapipe as mp
import winsound

def play_audio(audio_file):
    winsound.PlaySound(audio_file, winsound.SND_ASYNC)

def detect_hands():
    # Initialize MediaPipe hands model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Define box regions
    box_regions = [(50, 50, 200, 200),   # Tom (top-left corner)
                   (400, 50, 550, 200),  # Snare (top-right corner)
                   (50, 300, 200, 450),  # Kick (bottom-left corner)
                   (400, 300, 550, 450)] # Hi-hat (bottom-right corner)

    # Define box names
    box_names = ['Tom', 'Snare', 'Kick', 'Hi-hat']

    # Define audio files
    audio_files = ['Song/Tom.mp3', 'Song/Snare.mp3', 'Song/Kick.mp3', 'Song/Hi-hat.mp3']

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip only the camera frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert the image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # Loop through each detected hand
            for hand_landmarks in results.multi_hand_landmarks:
                # Check if the hand is in a fist
                is_fist = all(lm.x < 0.2 for lm in hand_landmarks.landmark[:5])  # Check thumb, index, middle, ring, and pinky finger

                # Get landmarks of index and other fingers
                index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                other_fingers = [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                                 hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                                 hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                                 hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]]

                # Check if index finger is extended and other fingers are closed
                is_index_extended = index_finger.y < other_fingers[0].y
                is_other_fingers_closed = all(other_finger.y > index_finger.y for other_finger in other_fingers)

                # Draw index finger landmark
                index_x, index_y = int(index_finger.x * frame.shape[1]), int(index_finger.y * frame.shape[0])
                cv2.circle(frame, (index_x, index_y), 5, (255, 0, 0), -1)

                # Draw hand landmarks (skeletal framework)
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check if the index finger is within any box region and the hand is not in a fist
                if not is_fist and is_index_extended and is_other_fingers_closed:
                    for i, box in enumerate(box_regions):
                        x1, y1, x2, y2 = box
                        if x1 < index_x < x2 and y1 < index_y < y2:
                            # Display box name
                            cv2.putText(frame, box_names[i], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            # Play corresponding sound using winsound
                            play_audio(audio_files[i])
                            break

        # Draw boxes on the frame
        for box in box_regions:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Hand Tracking', frame)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_hands()
