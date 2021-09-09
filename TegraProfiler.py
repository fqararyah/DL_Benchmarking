from Settings import Settings
import subprocess
from time import sleep
import utils
import Settings
import datetime

class TegraProfiler:
    def __init__(self, settings_obj, device_name):
        self.settings_obj = Settings.Settings()
        self.device_name = device_name

    def profile(self):
        while utils.get_status_line(self.settings_obj) != 'stop':
            if utils.get_status_line(self.settings_obj) == 'start':
                continue
            profiling_description = utils.get_status_line(self.settings_obj)
            if profiling_description != 'invalid' and profiling_description != None:
                f = open(self.settings_obj.current_folder + self.device_name + '_' + profiling_description +'_ved_mem_pow_profs.txt', 'a')
                f.write('\n*********************\n' + str(profiling_description) + '\n*********************\n')
            p = subprocess.Popen(['tegrastats', '--logfile', self.settings_obj.current_folder + self.device_name  + \
                '_mem_pow_profs.txt', '--interval', '200'])
            
            while utils.get_status_line(self.settings_obj) != 'invalid':
                pass

            p.terminate() 