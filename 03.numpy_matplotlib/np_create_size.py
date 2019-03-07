import numpy as np

a = np.empty( (2,3))
b = np.empty( (2,3), dtype=np.int16)
c = np.zeros( (2,3))
d = np.ones( (2,3), dtype=np.float32)
e = np.full( (2,3,4), 255, dtype=np.uint8)
print(a, a.dtype, a.shape)
print(b, b.dtype, b.shape)
print(c, c.dtype, c.shape)
print(d, d.dtype, d.shape)
print(e, e.dtype, e.shape)