import cv2, numpy as np

# 왼쪽 오른쪽 사진 읽기
imgL = cv2.imread('../img/restaurant1.jpg') # train
imgR = cv2.imread('../img/restaurant2.jpg') # query
hl, wl = imgL.shape[:2]
hr, wr = imgR.shape[:2]

grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# SIFT 특징 검출기 생성 및 특징점 검출
descriptor = cv2.xfeatures2d.SIFT_create()
(kpsL, featuresL) = descriptor.detectAndCompute(imgL, None)
(kpsR, featuresR) = descriptor.detectAndCompute(imgR, None)
# BF 매칭기 생성 및 knn매칭
matcher = cv2.DescriptorMatcher_create("BruteForce")
matches = matcher.knnMatch(featuresR, featuresL, 2)

# 좋은 매칭점 선별
good_matches = []
for m in matches:
    if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
        good_matches.append(( m[0].trainIdx, m[0].queryIdx))

# 좋은 매칭점이 4개 이상 원근 변환 행렬 구하기
if len(good_matches) > 4:
    ptsL = np.float32([kpsL[i].pt for (i, _) in good_matches])
    ptsR = np.float32([kpsR[i].pt for (_, i) in good_matches])
    mtrx, status = cv2.findHomography(ptsR,ptsL, cv2.RANSAC, 4.0)
    #원근 변환 행렬로 오른쪽 사진을 원근 변환, 결과 이미지 크기는 사진 2장 크기
    panorama = cv2.warpPerspective(imgR, mtrx, (wr + wl, hr))
    # 왼쪽 사진을 원근 변환한 왼쪽 영역에 합성
    panorama[0:hl, 0:wl] = imgL
else:
    panorama = imgL
cv2.imshow("Image Left", imgL)
cv2.imshow("Image Right", imgR)
cv2.imshow("Panorama", panorama)
cv2.waitKey(0)                        