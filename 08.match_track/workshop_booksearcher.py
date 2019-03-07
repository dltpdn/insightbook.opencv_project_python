import cv2 , glob, numpy as np

# 검색 설정 변수 ---①
ratio = 0.7
MIN_MATCH = 10
# ORB 특징 검출기 생성 ---②
detector = cv2.ORB_create()
# Flann 매칭기 객체 생성 ---③
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1)
search_params=dict(checks=32)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

# 책 표지 검색 함수 ---④
def serch(img):
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    
    results = {}
    # 책 커버 보관 디렉토리 경로 ---⑤
    cover_paths = glob.glob('../img/books/*.*')
    for cover_path in cover_paths:
        cover = cv2.imread(cover_path)
        cv2.imshow('Searching...', cover) # 검색 중인 책 표지 표시 ---⑥
        cv2.waitKey(5)
        gray2 = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
        kp2, desc2 = detector.detectAndCompute(gray2, None) # 특징점 검출 ---⑦
        matches = matcher.knnMatch(desc1, desc2, 2) # 특징점 매칭 ---⑧
        # 좋은 매칭 선별 ---⑨
        good_matches = [m[0] for m in matches \
                    if len(m) == 2 and m[0].distance < m[1].distance * ratio]
        if len(good_matches) > MIN_MATCH: 
            # 좋은 매칭점으로 원본과 대상 영상의 좌표 구하기 ---⑩
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])
            # 원근 변환 행렬 구하기 ---⑪
            mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # 원근 변환 결과에서 정상치 비율 계산 ---⑫
            accuracy=float(mask.sum()) / mask.size
            results[cover_path] = accuracy
    cv2.destroyWindow('Searching...')
    if len(results) > 0:
        results = sorted([(v,k) for (k,v) in results.items() \
                    if v > 0], reverse=True)
    return results

cap = cv2.VideoCapture(0)
qImg = None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('No Frame!')
        break
    h, w = frame.shape[:2]
    # 화면에 책을 인식할 영역 표시 ---⑬
    left = w // 3
    right = (w // 3) * 2
    top = (h // 2) - (h // 3)
    bottom = (h // 2) + (h // 3)
    cv2.rectangle(frame, (left,top), (right,bottom), (255,255,255), 3)
    
    # 거울 처럼 보기 좋게 화면 뒤집어 보이기
    flip = cv2.flip(frame,1)
    cv2.imshow('Book Searcher', flip)
    key = cv2.waitKey(10)
    if key == ord(' '): # 스페이스-바를 눌러서 사진 찍기
        qImg = frame[top:bottom , left:right]
        cv2.imshow('query', qImg)
        break
    elif key == 27 : #Esc
        break
else:
    print('No Camera!!')
cap.release()

if qImg is not None:
    gray = cv2.cvtColor(qImg, cv2.COLOR_BGR2GRAY)
    results = serch(qImg)
    if len(results) == 0 :
        print("No matched book cover found.")
    else:
        for( i, (accuracy, cover_path)) in enumerate(results):
            print(i, cover_path, accuracy)
            if i==0:
                cover = cv2.imread(cover_path)
                cv2.putText(cover, ("Accuracy:%.2f%%"%(accuracy*100)), (10,100), \
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow('Result', cover)
cv2.waitKey()
cv2.destroyAllWindows()
