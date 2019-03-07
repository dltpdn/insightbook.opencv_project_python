import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10)
f1 = x * 5
f2 = x **2
f3 = x **2 + x*2

plt.plot(x,'r--')   # 빨강색 이음선
plt.plot(f1, 'g.')  # 초록색 점
plt.plot(f2, 'bv')  # 파랑색 역 삼각형
plt.plot(f3, 'ks' ) # 검정색 사각형
plt.show()