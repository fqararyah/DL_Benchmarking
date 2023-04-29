from xml import dom
from xml.dom.minidom import DOMImplementation
import numpy as np
import sys
from models_archs import utils

utils.NET_PREFIX = 'resnet50'
utils.set_globals(utils.NET_PREFIX, utils.NET_PREFIX)

to_compare_layer_index = 7

VERY_DIFF_THRESHOLD = 2

ref = ''
if(len(sys.argv) > 1):
    to_compare_layer_index = int(sys.argv[1])
if(len(sys.argv) > 2):
    ref = sys.argv[2]

domain_file = './scratch_out/ofms_{}'.format(to_compare_layer_index)
if len(ref)>0:
    domain_file += '_' + ref + '.txt'
else:
    domain_file += '.txt'

range_file = './{}/fms/ofms_{}.txt'.format(utils.NET_PREFIX, to_compare_layer_index)

model_dag = utils.read_model_dag()

to_compare_layer_specs = model_dag[to_compare_layer_index]
to_compare_layer_ofms_shape = to_compare_layer_specs['ofms_shape']

ofms_hw = to_compare_layer_ofms_shape[1] * to_compare_layer_ofms_shape[2]
ofms_w = to_compare_layer_ofms_shape[2]
domain = np.loadtxt(domain_file).astype(np.int8)
rng = np.loadtxt(range_file).astype(np.int8)

# sum = 0
# for i in range(rng.size):
#     if (int(domain[i]) - rng[i] > 1 or int(domain[i]) - rng[i] < -1 ) and rng[i] != -128 and domain[i] != -128:
#         sum += int(domain[i]) - rng[i]

# print(sum)
print('(calculated, expected)')

sum = 0
cnt1=0
cnt2=0
cnt3=0
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
        d = int(i / ofms_hw)
        h = int((i % ofms_hw) / ofms_w)
        w = int(i % ofms_w) 
        position = (d, h, w)
        if (int(domain[i]) - rng[i] < -VERY_DIFF_THRESHOLD or int(domain[i]) - rng[i] > VERY_DIFF_THRESHOLD):# and (h != 0 and h != 55):
            diff_locs[position] = (domain[i], rng[i])
            #print(domain[i], rng[i])
            cnt3 += 1
        #    print(domain[i], rng[i])

# for key, val in diff_map.items():
#     print(key, val)
#     print('***************')

count = 0
for key, val in diff_locs.items():
    print(key, val)
    print('***************')
    count += 1
    if count > 100:
        break

print('max= ', np.max( np.abs(domain - rng)))
print(sum)
print(sum/rng.size)
print('equal: ', cnt1)
print('different: ', cnt2)
print('very different: ', cnt3)