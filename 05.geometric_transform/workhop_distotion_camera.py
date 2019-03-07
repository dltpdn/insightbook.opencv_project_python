import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
rows, cols = 240, 320
map_y, map_x = np.indices((rows, cols), dtype=np.float32)

# 거울 왜곡 효과 
map_mirrorh_x,map_mirrorh_y = map_x.copy(), map_y.copy() 
map_mirrorv_x,map_mirrorv_y = map_x.copy(), map_y.copy()    
## 좌우 대칭 거울 좌표 연산
map_mirrorh_x[: , cols//2:] = cols - map_mirrorh_x[:, cols//2:]-1
## 상하 대칭 거울 좌표 연산
map_mirrorv_y[rows//2:, :] = rows - map_mirrorv_y[rows//2:, :]-1

# 물결 효과
map_wave_x, map_wave_y = map_x.copy(), map_y.copy()
map_wave_x = map_wave_x + 15*np.sin(map_y/20)
map_wave_y = map_wave_y + 15*np.sin(map_x/20)    


# 렌즈 효과
## 렌즈 효과, 중심점 이동
map_lenz_x = 2*map_x/(cols-1)-1
map_lenz_y = 2*map_y/(rows-1)-1
## 렌즈 효과, 극좌표 변환
r, theta = cv2.cartToPolar(map_lenz_x, map_lenz_y)
r_convex = r.copy()
r_concave = r
## 볼록 렌즈 효과 매핑 좌표 연산
r_convex[r< 1] = r_convex[r<1] **2  
print(r.shape, r_convex[r<1].shape)
## 오목 렌즈 효과 매핑 좌표 연산
r_concave[r< 1] = r_concave[r<1] **0.5
## 렌즈 효과, 직교 좌표 복원
map_convex_x, map_convex_y = cv2.polarToCart(r_convex, theta)
map_concave_x, map_concave_y = cv2.polarToCart(r_concave, theta)
## 렌즈 효과, 좌상단 좌표 복원
map_convex_x = ((map_convex_x + 1)*cols-1)/2
map_convex_y = ((map_convex_y + 1)*rows-1)/2
map_concave_x = ((map_concave_x + 1)*cols-1)/2
map_concave_y = ((map_concave_y + 1)*rows-1)/2

while True:
    ret, frame = cap.read()
    # 준비한 매핑 좌표로 영상 효과 적용
    mirrorh=cv2.remap(frame,map_mirrorh_x,map_mirrorh_y,cv2.INTER_LINEAR)
    mirrorv=cv2.remap(frame,map_mirrorv_x,map_mirrorv_y,cv2.INTER_LINEAR)
    wave = cv2.remap(frame,map_wave_x,map_wave_y,cv2.INTER_LINEAR, \
                    None, cv2.BORDER_REPLICATE)
    convex = cv2.remap(frame,map_convex_x,map_convex_y,cv2.INTER_LINEAR)
    concave = cv2.remap(frame,map_concave_x,map_concave_y,cv2.INTER_LINEAR)
    # 영상 합치기
    r1 = np.hstack(( frame, mirrorh, mirrorv))
    r2 = np.hstack(( wave, convex, concave))
    merged = np.vstack((r1, r2))

    cv2.imshow('distorted',merged)
    if cv2.waitKey(1) & 0xFF== 27:
        break
cap.release
cv2.destroyAllWindows()