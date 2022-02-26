from cgitb import reset
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_hand = mp.solutions.hands
hand = mp_hand.Hands()

mp_draw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hand.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_lm in result.multi_hand_landmarks:
            for id, lm in enumerate(hand_lm.landmark):
                hight, width, channel = img.shape
                x, y = int(lm.x*width), int(lm.y*hight)
                print(id, x, y)

        mp_draw.draw_landmarks(img, hand_lm)

    cv2.imshow('Image', img)
    cv2.waitKey(1)