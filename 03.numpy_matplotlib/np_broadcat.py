import numpy as np

mylist = list(range(10))
print(mylist)

for i in range(len(mylist)):
    mylist[i] = mylist[i] + 1
print(mylist)