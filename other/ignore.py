import numpy as np

a = np.random.randint(low=0, high=127,size=(2,3,4,5))
print(a)
print('************')
a = np.reshape(a, (2,60))
print(a)
print('************')
print('************')
print('************')