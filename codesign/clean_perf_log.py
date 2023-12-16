import math

perf_log_file = 'fibha_perf_v2.txt'

model_cat = ''
models_perfs = {}
model_id = ''
runs_per_model = {}
with open(perf_log_file, 'r') as f:
    for line in f:
        line = line.replace(' ', '').replace('\n', '')
        if '.txt*************' in line:
            splits = line.split('/')
            if model_cat not in line:
                continue
            for split in splits:
                if '_configs.txt*************' in split:
                    file_name = split.split('_configs.txt*************')[0]
                    model_id = file_name.split('_')[-1]
                    if model_id.isnumeric():
                        model_id = int(model_id)
                        runs_per_model[model_id] = 0
                        models_perfs[model_id] = 0

        elif line.isnumeric() and model_id in models_perfs:
            models_perfs[model_id] += int(line)
            runs_per_model[model_id] += 1

# print(models_perfs)
for i in range(0, 1000):
    if i in models_perfs:
        print("%.3f" % ((models_perfs[i] / runs_per_model[i]) / 1000))
    # else:
    #     print('DNE')
