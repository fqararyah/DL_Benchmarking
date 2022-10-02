from xml import dom
from xml.dom.minidom import DOMImplementation
import numpy as np

from models_archs import utils

layer_indx = 7

layers_ofms_shape = utils.read_layers_output_shapes()

#layers_ofms_shape = {0: (32, 112, 112), 3: (16, 112, 112), 6: (24, 56, 56), 4: (96, 112, 112), 5: (96, 56, 56)}
domain_file = './scratch_out/ofms_{}.txt'.format(layer_indx)
range_file = './fms/fms_{}_{}_{}_{}.txt'.format(layer_indx + 1 if layer_indx > 0 else 2, layers_ofms_shape[layer_indx].depth, layers_ofms_shape[layer_indx].height,\
    layers_ofms_shape[layer_indx].width)

ofms_hw = layers_ofms_shape[layer_indx].height * layers_ofms_shape[layer_indx].width
domain = np.loadtxt(domain_file).astype(np.int8)
rng = np.loadtxt(range_file).astype(np.int8)

# sum = 0
# for i in range(rng.size):
#     if (int(domain[i]) - rng[i] > 1 or int(domain[i]) - rng[i] < -1 ) and rng[i] != -128 and domain[i] != -128:
#         sum += int(domain[i]) - rng[i]

# print(sum)
# print(sum/rng.size)

sum = 0
cnt1=0
cnt2=0
diff_map = {}
diff_locs = {}
for i in range(rng.size):
    if int(domain[i]) - rng[i] != 0:
        sum += abs(int(domain[i]) - rng[i])
        if domain[i] not in diff_map:
            diff_map[domain[i]] = {}
        if rng[i] not in diff_map[domain[i]]:    
            diff_map[domain[i]][rng[i]] = 0
        diff_map[domain[i]][rng[i]] += 1
    
    if domain[i] == rng[i]:
        cnt1 += 1
    elif int(domain[i]) - rng[i] !=0:
        cnt2 +=1
        if int(i / ofms_hw) not in diff_locs:
            diff_locs[ int(i / ofms_hw)] =0
        diff_locs[int(i / ofms_hw)] += 1
        #if int(domain[i]) - rng[i] > 10 or int(domain[i]) - rng[i] < -10:
        #    print(domain[i], rng[i])

for key, val in diff_map.items():
    print(key, val)
    print('***************')

for key, val in diff_locs.items():
    print(key, val)
    print('***************')

print(sum)
print(sum/rng.size)
print('equal: ', cnt1)
print('different: ', cnt2)