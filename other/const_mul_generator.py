import random

for i in range(64):
    #print('res[i] += tmpb[i][' + str(i) + '] *' + str (random.randrange(256)) + ';')
    print('tmp += tmpb[i][' + str(i) + '] <<' + str (random.randrange(9)) + ';')