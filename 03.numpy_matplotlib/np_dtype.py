import numpy as np

a = np.arange(10)
print(a, a.dtype)

b = a.astype('float32')
print(b, b.dtype)

c = np.uint8(b) 
print(c, c.dtype)

d = c.astype(np.float64)
print(d, d.dtype)