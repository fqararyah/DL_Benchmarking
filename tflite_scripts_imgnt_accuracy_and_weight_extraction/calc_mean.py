import numpy as np

arr = np.loadtxt('./eff_b0/fms/fms_conv2d_2_mul_1_2_32_112_112.txt')

arr2 = arr[:112*112]
arr2 = 0.26547086238861084 * (arr2 + 127)

print(np.mean(arr2) / 0.060017239302396774 - 123)

mean2 = np.mean(arr[:112*112])
mean2 = 0.26547086238861084 * (mean2 + 127) / 0.060017239302396774 - 123

print(mean2)
