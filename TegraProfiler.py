import Profiler
import subprocess
from time import sleep

class TegraProfiler(Profiler):
    def __init__(self, settings_obj, device_name):
        super().__init__(self, settings_obj, device_name)

    def profile(self):
        while super().get_status_line() != 'stop':
            profiling_description = super().get_status_line()
            p = subprocess.Popen(['tegrastats', '--logfile', self.device_name + '_profs.txt', '--interval', '100'])
            sleep(0.5)
            p.terminate()
            with open(self.settings_obj.current_folder + self.device_name, 'a') as f:
                f.write('\n*********************\n' + profiling_description + '\n*********************\n')
        