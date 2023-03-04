weights = [18, 28, 13, -7, -127, 11, 18, 33, 13]
fms = [-122, -123, -128,-128, -128 ,-128, -118, -116, -113]

res = 0

for i in range(9):
    res += (fms[i] + 128) * weights[i]
res += 9339 

scale_w_i_o = int(0.017232311889529228 * (2**32))


print(res)
res = ( (int(res * scale_w_i_o + (2**32)) >> 32 ) - 128)
print(res)

print(500000000>>32)