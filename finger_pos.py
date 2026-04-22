import pyautogui
import time

print("--- 좌표 찾기 도구를 시작합니다 ---")
print("좌표를 알고 싶은 곳에 마우스를 올리고 3초만 기다리세요.")
print("종료하려면 Ctrl + C를 누르세요.\n")

try:
    while True:
        # 현재 마우스의 X, Y 좌표를 가져옵니다.
        x, y = pyautogui.position()
        
        # 화면에 한 줄로 깔끔하게 출력 (값이 계속 변하면서 업데이트됨)
        position_str = f"현재 마우스 위치: X={x:4d}, Y={y:4d}"
        print(position_str, end="\r") 
        
        time.sleep(0.1) # 너무 빨리 출력되지 않게 약간의 휴식
except KeyboardInterrupt:
    print("\n\n좌표 찾기를 종료합니다.")