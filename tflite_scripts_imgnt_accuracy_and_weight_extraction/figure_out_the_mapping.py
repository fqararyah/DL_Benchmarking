from cv2 import sort
from dependencies import value
import numpy as np

domain_file = './scratch_out/ofms.txt'
range_file = './fms/fms_4_16_112_112.txt'

domain = np.loadtxt(domain_file).astype(np.int32)
range = np.loadtxt(range_file).astype(np.int8)

mapping_map = {}

indx=0
for element in range:
    if indx >= 112 * 112:
        break
    if element not in mapping_map:
        mapping_map[element] = []
    mapping_map[element].append(domain[indx])
    indx += 1

keys, vals = mapping_map.keys(), mapping_map.values()
keys,vals = zip(*sorted(zip(keys, vals)))

for key, val in zip(keys, vals):
    print(key, ':' , min(val), max(val), max(val) - min(val))