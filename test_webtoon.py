import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import webbrowser
import pyautogui
import time

# [단계 1] 제어할 웹툰 URL 입력
url = input("보실 웹툰의 URL을 입력하고 엔터를 누르세요: ")
webbrowser.open(url)

# [단계 2] 모델 및 기본 설정
actions = ['scroll_down', 'scroll_up', 'next_ep', 'prev_ep', 'menu']
model = load_model('models/webtoon_model.h5') 

seq_length = 30
seq = []
action_seq = [] 
last_action_time = 0 
click_locked = False 

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
window_name = 'AI_Webtoon_Controller'

# [핵심 수정] 창을 만들고 '항상 위' 속성을 부여합니다.
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1) # <--- 이 줄이 '항상 위' 고정!

cv2.resizeWindow(window_name, 320, 240)
cv2.moveWindow(window_name, 1500, 800) 

print("\n--- AI 컨트롤러 가동 중 (창이 항상 위에 유지됩니다) ---")

while cap.isOpened():
    ret, img = cap.read()
    if not ret: break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
            angle = np.degrees(angle)

            d = np.concatenate([joint.flatten(), angle])
            seq.append(d)
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length: continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            y_pred = model.predict(input_data, verbose=0).squeeze()
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            # 로그 출력
            print(f"감지: {actions[i_pred]} ({conf:.2f}) | 잠금: {'ON' if click_locked else 'OFF'}    ", end='\r')

            if conf > 0.96:
                action_seq.append(actions[i_pred])
            else:
                action_seq.append('nothing')

            if len(action_seq) >= 25:
                recent = action_seq[-25:]
                most_common = max(set(recent), key=recent.count)
                count = recent.count(most_common)
                current_time = time.time()

                if current_time - last_action_time > 3.0 and most_common != 'nothing':
                    
                    if most_common in ['scroll_down', 'scroll_up']:
                        if count >= 12:
                            print(f"\n[작동] {most_common} 스크롤")
                            for _ in range(10):
                                pyautogui.scroll(-65 if most_common == 'scroll_down' else 65)
                                time.sleep(0.01)
                            click_locked = False 
                            last_action_time = current_time

                    elif most_common in ['next_ep', 'prev_ep', 'menu']:
                        if count >= 20 and not click_locked:
                            if most_common == 'next_ep': target = (1646, 143)
                            elif most_common == 'prev_ep': target = (1452, 141)
                            else: target = (222, 145)
                            
                            print(f"\n[작동] {most_common} 클릭 -> 대기 위치(1598, 433) 이동")
                            pyautogui.moveTo(target[0], target[1], duration=0.6)
                            time.sleep(0.8)
                            pyautogui.click()
                            time.sleep(0.3)
                            pyautogui.moveTo(1598, 433, duration=0.5) 
                            
                            click_locked = True 
                            last_action_time = current_time

                cv2.putText(img, f'{actions[i_pred].upper()} {int(conf*100)}%', (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow(window_name, img)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()