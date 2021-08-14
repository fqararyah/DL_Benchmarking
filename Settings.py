
import os

class Settings:
    def __init__(self):
        self.current_folder = os.path.dirname(__file__) + '/'
        with open(self.current_folder + 'settings.txt', 'r') as f:
            for line in f:
                line = line.replace(' ', '').replace('\n', '')
                splits = line.split(':')
                if splits[0] == 'delimiter':
                    self.delimiter = line[line.index(':') + 1:]
                elif splits[0] == 'end_of_file':
                    self.end_of_file = splits[1]
                elif splits[0] == 'global_setting_keyword':
                    self.global_setting_keyword = splits[1]
                elif splits[0] == 'networks_file':
                    self.networks_file = self.current_folder + splits[1]
                elif splits[0] == 'batch_sizes_file':
                    self.batch_sizes_file = self.current_folder + splits[1]
                elif splits[0] == 'metrics_file':
                    self.metrics_file = self.current_folder + 'out/' + splits[1]
                elif splits[0] == 'input_dims_file':
                    self.input_dims_file = self.current_folder + splits[1]
                elif splits[0] == 'precisions_file':
                    self.precisions_file = self.current_folder + splits[1]
                elif splits[0] == 'tflite_folder':
                    self.tflite_folder = self.current_folder + splits[1]
                elif splits[0] == 'num_classes_file':
                    self.num_classes_file = self.current_folder + splits[1]
                elif splits[0] == 'status_file':
                    self.status_file_name = self.current_folder + splits[1]
                elif splits[0] == 'device_name':
                    self.device_name = splits[1]
                elif splits[0] == 'power_profile':
                    self.power_profile = int(splits[1])
                    
