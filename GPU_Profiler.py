from Settings import Settings
import subprocess
from time import sleep
import utils
import Settings
import datetime

class GPU_Profiler:
    def __init__(self, settings_obj, device_name):
        self.settings_obj = Settings.Settings()
        self.device_name = device_name

    def profile(self):
        while utils.get_status_line(self.settings_obj) != 'stop':
            profiling_description = utils.get_status_line(self.settings_obj)
            with open(self.settings_obj.current_folder + self.device_name  +'_mem_pow_profs.txt', 'a') as f:
                f.write('\n*********************\n' + str(profiling_description) + '\n*********************\n')
            f = open(self.settings_obj.current_folder + self.device_name  +'_mem_pow_profs.txt', 'a')
            p1 = subprocess.Popen(['nvidia-smi', '--query-gpu=power.draw', '--format=csv', '--loop-ms=20'], stdout=f)
            p2 = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.used', '--format=csv', '--loop-ms=20'], stdout=f)
            sleep(0.4)
            p1.terminate() 
            p2.terminate() 