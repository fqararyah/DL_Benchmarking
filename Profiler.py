from TegraProfiler import TegraProfiler
import Settings
import os

class Profiler:
    def __init__(self, settings_obj, device_name):
        self.settings_obj = settings_obj
        self.device_name = device_name
        self.profiling_description = self.get_status_line()

    def profile(self):
        pass
    
    def get_status_line(self):
        with open (self.settings_obj.status_file_name, 'r') as f:
            for line in f:
                line = line.replace(' ', '').replace('\n', '')
                self.profiling_description = line

settings_obj = Settings.Settings()
device_name = settings_obj.device_name.lower()
if settings_obj.power_profile  == 1:
    if 'tegra' in device_name or 'jetson' in device_name:
        profiler = TegraProfiler(settings_obj, device_name)
        profiler.profile()