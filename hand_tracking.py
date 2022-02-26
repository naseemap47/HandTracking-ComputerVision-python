import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mp_hand = mp.solutions.hands
hand = mp_hand.Hands()

mp_draw = mp.solutions.drawing_utils

p_time = 0
c_time = 0

while True:
    success, img = cap.read()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hand.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_lm in result.multi_hand_landmarks:
            for id, lm in enumerate(hand_lm.landmark):
                hight, width, channel = img.shape
                x, y = int(lm.x*width), int(lm.y*hight)
                if id==8:
                    cv2.circle(
                        img, (x,y), 8,
                        (255,0,255), cv2.FILLED
                    )

            mp_draw.draw_landmarks(img, hand_lm, mp_hand.HAND_CONNECTIONS)


    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.putText(
        img, str(int(fps)), (10,70),
        cv2.FONT_HERSHEY_PLAIN, 3,
        (255,0,255), 5
    )


    cv2.imshow('Image', img)
    cv2.waitKey(1)