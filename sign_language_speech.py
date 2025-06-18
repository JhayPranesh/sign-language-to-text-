import cv2
import mediapipe as mp
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import string

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

X, y = [],[]
label_map = {char: idx for idx, char in enumerate(string.ascii_uppercase)}
inverse_label_map = {v: k for k, v in label_map.items()}

print("\n[INFO] Starting sign language recorder")
print("[INFO] Press 'A' to 'Z' keys to record gestures.")
print("[INFO] Press 'T' to train & save model. Press 'Q' to quit.\n")

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    key = cv2.waitKey(1) & 0xFF  # Proper key capture
    current_label = None

    if 65 <= key <= 90:  # A-Z
        current_label = chr(key)
    elif key == ord('t'):
        print("[INFO] Training model...")

        if not X:
            print("⚠️ No data recorded. Please record hand gestures first.")
            continue

        X_np = np.array(X)
        y_np = to_categorical(y, num_classes=26)

        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_np.shape[1],)),
            Dense(64, activation='relu'),
            Dense(26, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_np, y_np, epochs=20, batch_size=16)

        model.save("sign_model.h5")
        print("✅ Model trained and saved as 'sign_model.h5'")
        break

    elif key == ord('q'):
        print("[INFO] Quitting without training.")
        break


    if results.multi_hand_landmarks:
     for handLms in results.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
        landmarks = []
        for lm in handLms.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        if current_label:
            X.append(landmarks)
            y.append(label_map[current_label])
            print(f"[+] Recorded gesture for '{current_label}' | Total: {len(X)} samples")
else:
    print("[INFO] No hand detected")



    cv2.imshow("Sign Language Recorder", frame)

cap.release()
cv2.destroyAllWindows()
