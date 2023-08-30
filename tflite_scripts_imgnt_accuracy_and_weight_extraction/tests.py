import numpy as np

arr = np.random.randint(low=0, high=255,size=(3,4,4))

#sub_arr = arr[:,0:3,0:3]
print(arr)

arr = np.pad(arr, ((0,0),(0,1),(0,1)), mode='constant', constant_values=-1)

print(arr)
