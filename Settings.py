

class Settings:
    def __init__(self):
        with open('settings.txt', 'r') as f:
            for line in f:
                line = line.replace(' ', '').replace('\n', '')
                splits = line.split(':')
                if splits[0] == 'networks_file':
                    self.networks_file = splits[1]
                elif splits[0] == 'batch_sizes_file':
                    self.batch_sizes_file = splits[1]
                elif splits[0] == 'delimiter':
                    self.delimiter = splits[1]
                elif splits[0] == 'end_of_file':
                    self.end_of_file = splits[1]
                elif splits[0] == 'global_setting_keyword':
                    self.global_setting_keyword = splits[1]
