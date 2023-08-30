import numpy as np

# a = np.random.randint(low=0, high=127,size=(2,3,4,5))

# print(a)
# print('************')
# a = np.reshape(a, (2,60))
# print(a)
# print('************')
# print('************')
# print('************')

# nwe_shape = (a.shape[-1],) + a.shape[:-1]
# print(a.shape)
# print(nwe_shape)

a = np.random.randint(low=0, high=127,size=(2,3,4))
print(a)
a = np.transpose(a, (-1, 0, 1))
print(a)