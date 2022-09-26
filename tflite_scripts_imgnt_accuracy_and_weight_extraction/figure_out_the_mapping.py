from cv2 import sort
from dependencies import value
import numpy as np

layer_indx = 4
layers_ofms_shape = {3: (16, 112, 112), 6: (24, 56, 56), 4: (96, 112, 112)}
domain_file = './scratch_out/ofms_{}.txt'.format(layer_indx)
range_file = './fms/fms_{}_{}_{}_{}.txt'.format(layer_indx + 1, layers_ofms_shape[layer_indx][0], layers_ofms_shape[layer_indx][1],\
    layers_ofms_shape[layer_indx][2])

layers_scale_ifms = {3: 0.0235294122248888, 6: 0.0235294122248888, 4:0.3023846447467804 }
layers_scale_weights = {3: 0.02902807, 6: 0.01043679, 4: 0.00100364}
layers_scale_ofms = {3: 0.3023846447467804,  6: 0.1985088586807251, 4:0.0235294122248888} 
layers_ofms_zero_point = {3: 6, 6: 5, 4: 128}

domain = np.loadtxt(domain_file).astype(np.int32)
range = np.loadtxt(range_file).astype(np.int8)

mapping_map = {}

indx=0
for element in range:
    if indx >= layers_ofms_shape[layer_indx][1] * layers_ofms_shape[layer_indx][2]:
        break
    if element not in mapping_map:
        mapping_map[element] = []
    mapping_map[element].append(domain[indx])
    indx += 1

keys, vals = mapping_map.keys(), mapping_map.values()
keys,vals = zip(*sorted(zip(keys, vals)))

for key, val in zip(keys, vals):
    print(key, ':' , min(val), max(val), max(val) - min(val), max(val) * (layers_scale_ifms[layer_indx] * \
         layers_scale_weights[layer_indx])
         / layers_scale_ofms[layer_indx] - layers_ofms_zero_point[layer_indx])

# a = np.random.randint(1, 128, (5,5)).astype(np.int8)
# a[0,0] = 128
# print(a)