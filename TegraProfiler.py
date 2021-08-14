from Settings import Settings
import subprocess
from time import sleep
import utils
import Settings

class TegraProfiler:
    def __init__(self, settings_obj, device_name):
        self.settings_obj = Settings.Settings()
        self.device_name = device_name

    def profile(self):
        while utils.get_status_line(self.settings_obj) != 'stop':
            profiling_description = utils.get_status_line(self.settings_obj)
            p = subprocess.Popen(['tegrastats', '--logfile', self.device_name + '_profs.txt', '--interval', '100'])
            sleep(0.5)
            p.terminate()
            with open(self.settings_obj.current_folder + self.device_name, 'a') as f:
                f.write('\n*********************\n' + str(profiling_description) + '\n*********************\n')
        