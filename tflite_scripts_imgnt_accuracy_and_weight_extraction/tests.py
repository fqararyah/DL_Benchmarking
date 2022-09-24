import numpy as np

arr = np.random.randint(low=0, high=255,size=(3,224,224))

sub_arr = arr[:,0:3,0:3]
print(sub_arr.shape)