import cv2
import mediapipe as mp
import numpy as np
import time, os

# --------------------------------------------------------
# [단계 1] 기본 세팅: 수집할 동작과 촬영 시간 설정
# --------------------------------------------------------
# 우리가 사용할 5가지 웹툰 제어 동작으로 수정했습니다.
actions = ['scroll_down', 'scroll_up', 'next_ep', 'prev_ep', 'menu']
seq_length = 30      # LSTM 모델이 한 번에 바라볼 프레임 수 (연속된 30장의 사진을 하나의 동작으로 봄)
secs_for_action = 30 # 하나의 동작당 데이터를 수집할 시간 (30초 동안 셔터를 계속 누르고 있는 것과 같습니다)

# MediaPipe 손 인식 모델 초기화 (뼈대 추출용)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1, # 한 손만 인식하도록 설정
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 웹캠 켜기
cap = cv2.VideoCapture(0)

# 데이터가 섞이지 않도록 파일명에 붙일 고유 시간값 생성
created_time = int(time.time())
os.makedirs('dataset', exist_ok=True) # dataset 폴더가 없으면 만듭니다.

# --------------------------------------------------------
# [단계 2] 동작별로 순회하며 데이터 수집 시작
# --------------------------------------------------------
while cap.isOpened():
    # actions 리스트에 있는 5개 동작을 하나씩 차례대로 돕니다.
    for idx, action in enumerate(actions):
        data = [] # 추출된 뼈대 좌표를 담을 빈 바구니

        ret, img = cap.read()
        img = cv2.flip(img, 1) # 거울처럼 보이도록 좌우 반전

        # [기능 1] 준비 시간 주기 (3초 대기)
        # 동작을 시작하기 전, 화면에 어떤 동작을 할 차례인지 띄워주고 3초간 기다립니다. (포즈 잡을 시간)
        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000)

        start_time = time.time()

        # [기능 2] 본격적인 데이터 수집 (30초 동안)
        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img) # MediaPipe가 손가락 관절 위치를 찾습니다.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 화면에 손이 보이면 관절 좌표를 계산합니다.
            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility] # 21개 관절의 x,y,z 좌표 저장

                    # 관절 사이의 '각도'를 계산하는 수학적 연산 (이 각도 정보가 동작을 구분하는 핵심 키가 됩니다)
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] 
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] 
                    v = v2 - v1 
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) 
                    angle = np.degrees(angle) # 라디안을 우리가 아는 각도(도) 단위로 변환

                    # 각도 데이터 끝에 현재 동작의 정답지(idx: 0, 1, 2, 3, 4)를 붙여표기
                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)

                    # 관절 좌표(joint)와 각도+정답지(angle_label)를 한 줄로 쭉 펴서 연결
                    d = np.concatenate([joint.flatten(), angle_label])
                    data.append(d) # 바구니에 담기

                    # 화면에 관절 선 그려주기 (시각적 확인용)
                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'): # 중간에 q를 누르면 강제 종료
                break

        # --------------------------------------------------------
        # [단계 3] 수집된 데이터 편집 및 저장 (.npy 파일 생성)
        # --------------------------------------------------------
        data = np.array(data)
        print(action, data.shape) # 수집된 총 프레임 수 출력
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data) # 날것의 데이터 저장

        # [기능 3] 시퀀스(연속 동작) 데이터로 쪼개기
        # 카메라로 찍은 긴 영상을 30프레임짜리 짧은 움짤(gif) 여러 개로 자르는 작업입니다.
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape) # 쪼개진 시퀀스 덩어리의 개수 출력
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data) # 최종 학습용 데이터 저장
        
    break # 5개 동작 수집이 다 끝나면 while문을 빠져나와 카메라를 끕니다.

cap.release()
cv2.destroyAllWindows()