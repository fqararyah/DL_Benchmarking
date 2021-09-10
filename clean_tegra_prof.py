from posixpath import split
import Settings

settings_obj = Settings.Settings()

in_file = settings_obj.current_folder + 'jetson_nano_mem_pow_profs_mixed.txt'
out_file = settings_obj.current_folder + 'pow_mem_jetson_nano_224x224.txt'

power_dict = {}
with open(in_file, 'r') as f:
    current_key = ''
    for line in f:
        line = line.replace('\n', '')
        splits = line.split(' ')
        if len(splits) == 1:
            if '*' not in splits[0] and len(splits[0]):
                current_key = splits[0]
                if current_key not in power_dict:
                    power_dict[current_key] = [{'VDD_SYS_GPU': 0,'VDD_SYS_SOC': 0, 'VDD_IN': 0, 'VDD_SYS_DDR': 0, 'RAM': 0.0}, 0]
        else:
            for i in range(len(splits)):
                if splits[i] in power_dict[current_key][0]:
                    power_dict[current_key][0][splits[i]] += int(splits[i + 1].split('/')[0])
            power_dict[current_key][1] += 1

for key, val in power_dict.items():
    for part_pow_cons_key, part_pow_cons_val in val[0].items():
        val[0][part_pow_cons_key] /= val[1]

power_cons_avg = 0
with open(out_file, 'w') as f:
    for key, val in power_dict.items():
        f.write(key + '\n')
        for part_pow_cons_key, part_pow_cons_val in val[0].items():
            f.write(part_pow_cons_key + ' ' + str(int(part_pow_cons_val)) + ' ')
            power_cons_avg += part_pow_cons_val
        f.write('\n')
    f.write('************************\nAverage power consumptio is: ' + power_cons_avg / len(power_dict))
