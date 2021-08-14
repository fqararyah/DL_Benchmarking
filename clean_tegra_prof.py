from posixpath import split
import Settings

settings_obj = Settings.Settings()

in_file = settings_obj.current_folder + 'jetson_tx2_profs_32x32.txt'
out_file = settings_obj.current_folder + 'jetson_tx2_pow_32x32.txt'

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
                    power_dict[current_key] = [{'VDD_SYS_GPU': 0.0,'VDD_SYS_SOC': 0.0, 'VDD_IN': 0.0, 'VDD_SYS_DDR': 0.0}, 0]
        else:
            for i in range(len(splits)):
                if splits[i] in power_dict[current_key][0]:
                    power_dict[current_key][0][splits[i]] += float(splits[i + 1].split('/')[0])
                    power_dict[current_key][1] += 1

for key, val in power_dict.items():
    for part_pow_cons_key, part_pow_cons_val in val[0].items():
        val[0][part_pow_cons_key] /= (val[1] + 1)

print(power_dict)