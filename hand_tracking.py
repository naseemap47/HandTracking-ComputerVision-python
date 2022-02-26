from cgitb import reset
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_hand = mp.solutions.hands
hand = mp_hand.Hands()

while True:
    success, img = cap.read()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hand.process(img_rgb)
    print(result.multi_hand_landmarks)

    cv2.imshow('Image', img)
    cv2.waitKey(1)