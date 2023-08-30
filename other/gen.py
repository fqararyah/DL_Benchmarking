
from math import trunc
import random

no_of_generated = 64

biggest_pow_of_2 = 128

max_itr = 3

powers_indices = {}
for i in range(no_of_generated):
    val = random.randrange(biggest_pow_of_2 * 2)
    if val == 0:
        continue
    #print(val)
    current_pow_of_2 = biggest_pow_of_2
    odd = False
    if val % 2 == 1:
        odd = True
        val -= 1
    
    itr = 0
    while current_pow_of_2 > 1 and itr < max_itr:
        if val >= current_pow_of_2:
            itr += 1
            if current_pow_of_2 not in powers_indices:
                powers_indices[current_pow_of_2] = []
            powers_indices[current_pow_of_2].append(i)
            val -= current_pow_of_2
        current_pow_of_2 /= 2

line = 'tmp = '
for power, indices in powers_indices.items():
    line += str(power) + ' * ('
    for indx in indices:
        line += 'tmpa[' + str(indx) + '] + '
    line = line[0:-2]
    line += ')\n + '
line = line[0: -2] + ';'
print(line)
