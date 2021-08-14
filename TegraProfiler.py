import subprocess
from time import sleep
import utils

class TegraProfiler:
    def __init__(self, settings_obj, device_name):
        self.settings_obj = settings_obj
        self.device_name = device_name
        self.profiling_description = utils.get_status_line()

    def profile(self):
        while utils.get_status_line() != 'stop':
            profiling_description = super().get_status_line()
            p = subprocess.Popen(['tegrastats', '--logfile', self.device_name + '_profs.txt', '--interval', '100'])
            sleep(0.5)
            p.terminate()
            with open(self.settings_obj.current_folder + self.device_name, 'a') as f:
                f.write('\n*********************\n' + profiling_description + '\n*********************\n')
        