import numpy as np

arr1 = np.loadtxt('comp1.txt')
arr2 = np.loadtxt('comp2.txt')

diff = np.abs(arr1 - arr2)

print('max', np.max(diff), 'at', np.argmax(diff))
print('average', np.average(diff))