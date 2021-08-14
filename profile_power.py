import Settings
from TegraProfiler import TegraProfiler

settings_obj = Settings.Settings()
device_name = settings_obj.device_name.lower()
if settings_obj.power_profile  == 1:
    if 'tegra' in device_name or 'jetson' in device_name:
        profiler = TegraProfiler(settings_obj, device_name)
        profiler.profile()