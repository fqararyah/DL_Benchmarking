import numpy as np

arr = np.loadtxt('./eff_b0/fms/fms_conv2d_1_a1_mul_1_32_112_112.txt')

arr2 = arr[-112*112:]
arr2 = 0.26547086238861084 * (arr2 + 127)

print(np.mean(arr2) / 0.060017239302396774 - 123)
