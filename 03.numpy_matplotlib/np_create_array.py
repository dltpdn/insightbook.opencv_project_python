import numpy as np

a = np.array([1,2,3,4])     # 정수를 갖는 리스트로 생성
b = np.array([[1,2,3,4],    # 2차원 리스트로 생성
              [5,6,7,8]])
c = np.array([1,2,3.14,4])  # 정수와 소수점이 혼재된 리스트
d = np.array([1,2,3,4], dtype=np.float64)   # dtype을 지정해서 생성

print(a, a.dtype, a.shape)  # --- ①
print(b, b.dtype, b.shape)  # --- ②
print(c, c.dtype, c.shape)  # --- ③
print(d, d.dtype, d.shape)  # --- ④