from ast import In
from xml import dom
from xml.dom.minidom import DOMImplementation
import numpy as np
import sys
from models_archs import utils

utils.NET_PREFIX = 'mob_v2'
utils.set_globals(utils.NET_PREFIX, utils.NET_PREFIX)

IN = 0
OUT = 1
IN_OUT = OUT 

to_compare_layer_index = 7

VERY_DIFF_THRESHOLD = 5

ref = ''
if(len(sys.argv) > 1):
    to_compare_layer_index = int(sys.argv[1])
if(len(sys.argv) > 2):
    ref = sys.argv[2]

model_dag = utils.read_model_dag()

domain_file = './scratch_out/ofms_{}.txt'.format(to_compare_layer_index)

layer_children = model_dag[to_compare_layer_index]["children"]
fused_with_add =  "add" in model_dag[layer_children[0]]["name"] and \
      model_dag[layer_children[0]]['id'] == to_compare_layer_index + 1
if ref == '' and fused_with_add:
    to_compare_layer_index = model_dag[layer_children[0]]["id"]
if len(ref)>0:
    range_file = './scratch_out/ofms_{}_ref.txt'.format(to_compare_layer_index)
else:
    if IN_OUT == IN:
        range_file = './{}/fms/ofms_{}.txt'.format(utils.NET_PREFIX, to_compare_layer_index)
    else:
        range_file = './{}/fms/ofms_{}.txt'.format(utils.NET_PREFIX, to_compare_layer_index)

print(domain_file, range_file)

to_compare_layer_specs = model_dag[to_compare_layer_index]

if IN_OUT == IN:
    to_compare_layer_ofms_shape = to_compare_layer_specs['ifms_shape']
else:
    to_compare_layer_ofms_shape = to_compare_layer_specs['ofms_shape']

print(to_compare_layer_ofms_shape)

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
for i in range(min(rng.size, domain.size)):
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
        position = (i, d, h, w)
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
    if key[1] == 3 or 1 == 1:
        print(key, val)
        print('***************')
        count += 1
    if count > 100:
        break

num_elements = cnt1 + cnt2

if(rng.size != domain.size):
    print("SIZE MIMATCH")
else:
    print('max= ', np.max( np.abs(domain - rng)))
    print(sum)
    print(sum/rng.size)
    print('equal: ', cnt1 / num_elements)
    print('different: ', cnt2 / num_elements)
    print('very different: ', cnt3 / num_elements)