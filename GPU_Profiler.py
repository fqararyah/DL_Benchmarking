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
            if utils.get_status_line(self.settings_obj) == 'start':
                continue
            profiling_description = utils.get_status_line(self.settings_obj)
            if profiling_description != 'invalid' and profiling_description != None and 'invalid' not in profiling_description:
                f = open(self.settings_obj.current_folder + self.device_name + '_' + profiling_description +'_ved_mem_pow_profs.txt', 'a')
                f.write('\n*********************\n' + str(profiling_description) + '\n*********************\n')
                p1 = subprocess.Popen(['nvidia-smi', '--query-gpu=power.draw', '--format=csv', '--loop-ms=5'], stdout=f)
                p2 = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.used', '--format=csv', '--loop-ms=5'], stdout=f)
            
            while utils.get_status_line(self.settings_obj) != 'invalid' and utils.get_status_line(self.settings_obj) != 'stop':
                pass
            p1.terminate() 
            p2.terminate()